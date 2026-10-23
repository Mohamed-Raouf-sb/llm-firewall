# Real-Time AI Firewall for LLM Prompt Injection and Jailbreak Detection

A small, deployable inspection proxy that scores prompts before forwarding them to an LLM. It supports a trained Hugging Face sequence classifier and a built-in rule baseline for local development. The baseline is **not** a substitute for a trained detector or upstream LLM security controls.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-api.txt
python -m uvicorn app.main:app --reload
```

If your shell has multiple Python installations, use the virtual environment's interpreter explicitly: `.venv/bin/python -m pip install -r requirements-api.txt` followed by `.venv/bin/python -m uvicorn app.main:app --reload`. This avoids accidentally running a global `uvicorn` launcher from a different Python installation.

Open `http://127.0.0.1:8000/` for the prompt inspector, `http://127.0.0.1:8000/chat` for the firewall-protected chatbot, or `http://127.0.0.1:8000/docs` for the API. The inspector shows detector decisions and scores. The chatbot sends messages through the firewall before forwarding allowed requests to an OpenAI-compatible LLM. Its API key remains server-side. With no model configured, the detector uses **rules mode**; with no upstream LLM configured, chat is disabled.

## Connect an upstream LLM

Set the upstream base URL to the provider's API base (for example, `https://api.openai.com/v1`), the model name, and its key in the server environment. The firewall scans user messages, returns `403` for blocked or review-required prompts, and forwards allowed requests. It replaces client-supplied system/developer messages with a trusted security system prompt before forwarding. This prompt adds defense in depth; it does not replace the detector. Upstream errors and HTTP status codes are returned to the caller, while connection failures return `502` and timeouts return `504`.

Copy `.env.example` to `.env`, enter your provider key in `UPSTREAM_API_KEY`, and start the API. The settings loader reads `.env` automatically:

```bash
cp .env.example .env
# Edit .env and add your private API key, then run:
.venv/bin/python -m uvicorn app.main:app --reload
```

`.env` is excluded by `.gitignore`; `.env.example` contains no secrets and is safe to commit. On Render, set the same values in the service's Environment settings instead of uploading a `.env` file.

Point an OpenAI-compatible client at `http://127.0.0.1:8000/v1` and use `/chat/completions` as usual. Keep the provider key in an environment variable; do not commit it to the project. The web chatbot is at `/chat` and sends its conversation through the same firewall endpoint.

## Deploy the app on Render

The model checkpoint is not committed to this GitHub repository. Upload the local model directory to a Hugging Face model repository first. From the project root, install the Hub CLI in your virtual environment and authenticate with a Hugging Face write token:

```bash
.venv/bin/python -m pip install huggingface_hub
.venv/bin/hf auth login
.venv/bin/hf upload r4ouf-sb44/prompt-firewall-mohamed-fix artifacts/prompt-firewall-mohamed-fix . --repo-type model
```

Create the Hugging Face model repository before uploading. A private repository is recommended. Do not put the Hugging Face token in this repository.

The repository uses a Dockerfile, so Render will build from that file; there are no separate Build Command or Start Command fields for this Docker service. The Dockerfile installs CPU-only inference dependencies and binds Uvicorn to Render's `PORT`. Push changes to `main` to trigger a rebuild, or select **Manual Deploy → Deploy latest commit**.

For the transformer checkpoint, choose a Render instance with at least 2 GB RAM. The Free plan has 512 MB and is too small for reliable transformer inference. The rules-only version needs less memory.

Add these environment variables under **Environment**:

| Key | Value |
|---|---|
| `MODEL_PATH` | `r4ouf-sb44/prompt-firewall-mohamed-fix` |
| `BLOCK_THRESHOLD` | `0.99228` |
| `REVIEW_THRESHOLD` | `0.75` |
| `HF_TOKEN` | A Hugging Face **read** token, only if the model repository is private |
| `UPSTREAM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `UPSTREAM_MODEL` | `openai/gpt-oss-20b` |
| `UPSTREAM_API_KEY` | Your Groq API key; mark it secret |

Keep `HF_TOKEN` marked secret in Render. For a public model repository, omit it. Save the changes and redeploy. After the first deploy downloads the checkpoint, open the service URL (the root page, not `/docs`); the interface should report `transformer` as the detector. Confirm this at `https://YOUR-SERVICE.onrender.com/health`, whose `detector` field should be `transformer`.

Example inspection request:

```bash
curl -s http://127.0.0.1:8000/v1/inspect \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hi, my name is Mohamed."}'
```

Example OpenAI-compatible request (inspection-only unless an upstream is configured):

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"example-model","messages":[{"role":"user","content":"Hello"}]}'
```

## Train with your data

Provide a CSV with columns `text` and `label`. Labels may be `0`/`safe`/`benign` or `1`/`malicious`/`injection`/`jailbreak`. Keep train and test examples independent; evaluation metrics are only meaningful when the test split represents attacks and benign traffic you expect in deployment.

To fetch and export the Neuralchemy `core` subset into those three split files (requires Hugging Face network access):

```bash
python -m pip install datasets
python scripts/download_neuralchemy.py
```

The exporter preserves the dataset's official train, validation, and test boundaries and writes only `text,label`. Train with those splits directly; do **not** run `prepare_data.py` on an already split file, since that script is for a single unsplit CSV.

```bash
python scripts/prepare_data.py --input data/raw/prompts.csv --output data/processed
python scripts/train.py --data-dir data/processed --output-dir artifacts/prompt-firewall
python scripts/evaluate.py --model artifacts/prompt-firewall --data data/processed/validation.csv --tune-threshold --target-recall 0.90 --output outputs/validation_metrics.json
THRESHOLD=$(python -c 'import json; print(json.load(open("outputs/validation_metrics.json"))["threshold"])')
python scripts/evaluate.py --model artifacts/prompt-firewall --data data/processed/test.csv --threshold "$THRESHOLD" --output outputs/test_metrics.json
```

Use validation data to select a threshold, then freeze that value before the final test evaluation. Do not tune on the test split. Test metrics are a held-out estimate, not a guarantee of production performance.

For training only, install the full stack with `pip install -r requirements.txt` (this includes PyTorch and Accelerate, required by Hugging Face `Trainer`). Training downloads the default lightweight model (`distilbert/distilbert-base-uncased`) from Hugging Face. Set `BASE_MODEL` or pass `--base-model` to use another compatible sequence-classification backbone. Install GPU-enabled PyTorch separately if needed. Do not upload confidential prompts or secrets to public services.

After training, set `MODEL_PATH` to the trained model directory and restart the API. The detector fails closed on missing/corrupt configured models unless `FAIL_OPEN=true` is explicitly set.

The recommended local artifact `artifacts/prompt-firewall-mohamed-fix` was fine-tuned from the hardened checkpoint with additional benign greetings and introductions, plus balanced replay examples from the original training split. It correctly scores `hi my name is mohamed.` as safe (malicious score `0.00068`) in a direct check. Start the API with:

```bash
MODEL_PATH=artifacts/prompt-firewall-mohamed-fix BLOCK_THRESHOLD=0.99228 REVIEW_THRESHOLD=0.75 .venv/bin/python -m uvicorn app.main:app --reload
```

The threshold `0.99228` was selected on validation data to target 90% recall; validation recall was 90.1% with a 0.49% false-positive rate (2 of 407 benign prompts). At this frozen threshold, held-out test recall was 89.5% with a 1.79% false-positive rate (7 of 390 benign prompts). See `outputs/validation_metrics_mohamed_fix.json` and `outputs/test_metrics_mohamed_fix.json`. These results apply to this dataset and are not a safety guarantee. Prompts at or above `REVIEW_THRESHOLD=0.75` are withheld from the upstream LLM and return `review_required`.

### Evaluation summary

The data is the `core` configuration of `neuralchemy/Prompt-injection-dataset` (4,391 train, 941 validation, and 942 test examples). The recommended checkpoint was fine-tuned from the hardened classifier using additional benign greetings and introductions plus balanced replay examples from the original training split. The untouched validation and test files were used for evaluation. At the validation-selected threshold:

| Split | Accuracy | Precision | Recall | F1 | False-positive rate | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---:|---|
| Validation (941 prompts) | 94.2% | 99.6% | 90.1% | 94.6% | 0.49% | 405, 2, 53, 481 |
| Test (942 prompts) | 93.1% | 98.6% | 89.5% | 93.8% | 1.79% | 383, 7, 58, 494 |

The test results show both remaining misses (58 malicious examples) and false alarms (7 benign examples). The dataset is one benchmark and may not represent the prompts, languages, encodings, or indirect attacks seen in a deployment. The reported softmax scores are not calibrated probabilities of harm. Treat the model as a defense-in-depth signal, not a standalone security boundary.

### How thresholds work

The model calculates a new malicious score for every prompt. `BLOCK_THRESHOLD` and `REVIEW_THRESHOLD` are fixed policy cutoffs for the running server; the score changes per prompt, while the cutoff stays fixed so decisions remain reproducible. The default `0.90` is set so the rules fallback's `0.92` injection score is blocked. For the recommended transformer checkpoint, the validation-selected cutoff is `0.99228`; set that explicitly when deploying that model. Thresholds do not transfer reliably between detector types or checkpoints. Lower cutoffs block more prompts, including more benign ones.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | unset | Local model directory or Hugging Face model repository ID |
| `BLOCK_THRESHOLD` | `0.90` | Malicious probability at/above which prompts are blocked; validate this cutoff for your detector |
| `REVIEW_THRESHOLD` | `0.75` | Malicious probability at/above which prompts receive `review` instead of `allow` |
| `FAIL_OPEN` | `false` | Continue with rules if configured model fails to load |
| `MAX_PROMPT_CHARS` | `20000` | Input size limit |
| `UPSTREAM_BASE_URL` | unset | OpenAI-compatible API base URL, e.g. `https://api.openai.com/v1` |
| `UPSTREAM_MODEL` | `gpt-4o-mini` | Model identifier sent to the upstream API |
| `UPSTREAM_API_KEY` | unset | Upstream bearer token; keep it secret and server-side |
| `UPSTREAM_TIMEOUT_SECONDS` | `30` | Upstream request timeout |

## API

- `GET /health`: service status, detector mode, configured model path, and active thresholds; use this to confirm the server loaded the intended model (rules mode means no transformer model is active)
- `POST /v1/inspect`: `{ "text": "..." }` → model-predicted class, both class scores, decision (`allow`, `review`, or `block`), configured thresholds, explanation, and inference latency. The class scores come from the model's softmax output; they are not calibrated probabilities of real-world harm. `allow` means only that the score is below the review threshold; it is not a guarantee of safety.
- `POST /v1/chat/completions`: OpenAI-compatible request; blocks flagged or review-required user messages and forwards allowed requests if upstream is configured, preserving streaming and compatible extra fields

When no upstream is configured, `/v1/chat/completions` does not generate an LLM answer; it only confirms that the request passed inspection. The API has no built-in authentication or rate limiting, so keep it on a trusted network or put an authenticated gateway in front of it before exposing it publicly.

This is a research prototype. Prompt injection detection is probabilistic: it can miss attacks and flag benign text. Use defense in depth, least-privilege tools, server-side authorization, and human review for consequential actions. Avoid logging prompt contents in production.

## Docker

```bash
docker build -t llm-firewall .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/models/prompt-firewall-mohamed-fix \
  -e BLOCK_THRESHOLD=0.99228 \
  -v "$PWD/artifacts:/models:ro" \
  llm-firewall
```

The local `artifacts/` directory is ignored by Git and must exist on the host before running the container. The image installs the full training requirements; this is convenient for the prototype but makes the image larger than a production runtime-only image.

### Docker Hub DNS troubleshooting

If host lookups succeed but Docker pulls time out, test the resolver with and without EDNS:

```bash
dig @<dns-server> registry-1.docker.io A +time=2 +tries=1
dig +noedns @<dns-server> registry-1.docker.io A +time=2 +tries=1
```

If the regular query times out but the `+noedns` query succeeds, the network DNS server may be incompatible with EDNS. Go enables EDNS by default and supports disabling it with `GODEBUG=netedns0=0`. For the Docker Snap systemd service, create an override:

```bash
sudo mkdir -p /etc/systemd/system/snap.docker.dockerd.service.d
printf '[Service]\nEnvironment=GODEBUG=netedns0=0\n' | sudo tee /etc/systemd/system/snap.docker.dockerd.service.d/override.conf
sudo systemctl daemon-reload
sudo snap restart docker.dockerd
sudo snap services docker
docker info
```

Wait until the service is `active` and `docker info` returns server details, then test the pull:

```bash
docker pull python:3.11-slim
```

To revert the override, remove `/etc/systemd/system/snap.docker.dockerd.service.d/override.conf`, run `sudo systemctl daemon-reload`, and restart the Snap daemon. Docker's `daemon.dns` setting configures DNS for containers; it does not change the host resolver used for registry lookups. The BuildKit warning about missing Git commit information is separate from DNS failures.

## Dataset sources

Place data you are authorized to use in `data/raw/`. Record source, license, collection date, and preprocessing in your report. Publicly available datasets can have licensing and distribution restrictions; review those before redistribution. This repository does not bundle third-party data. The saved evaluation metrics are specific to the reported local run and are not official leaderboard results.

## Project report

See [reports/project_report.md](reports/project_report.md) for the project design, evaluation results, limitations, deployment notes, and future work.
