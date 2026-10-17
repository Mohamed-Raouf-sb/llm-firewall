# Project Report: Real-Time AI Firewall for Prompt Injection Detection

**Project type:** Applied NLP and cybersecurity prototype  
**Implementation:** Python, PyTorch, Hugging Face Transformers, FastAPI  
**Report scope:** Current API implementation and saved evaluation artifacts

## Abstract

This project implements an intermediate inspection service for text sent to a large language model (LLM). A fine-tuned DistilBERT sequence classifier scores user prompts for prompt-injection or jailbreak characteristics. A policy layer uses configurable score thresholds to allow, review, or block requests. The FastAPI service also provides an OpenAI-compatible chat-completions route that can forward allowed requests to an upstream provider.

The saved held-out evaluation reports show 89.5% recall and a 1.79% false-positive rate at the selected blocking threshold. The model also produces false negatives and false positives, so the firewall is a prototype defense-in-depth component, not a guarantee that requests are safe.

## 1. Problem and objectives

LLM applications can receive instructions designed to override their intended behavior, reveal hidden instructions, or influence downstream actions. Static keyword filters can miss paraphrases and other semantically varied attacks. The project addresses this by scoring each incoming text prompt with a learned text classifier before the prompt reaches an upstream LLM.

The project objectives are to:

- Train a lightweight binary classifier for safe and malicious prompt examples.
- Place the classifier in a real-time HTTP inspection and forwarding service.
- Measure classification quality and expose threshold-based decisions.
- Document operational limitations and evaluation evidence.

## 2. System design

The request path has four stages:

1. FastAPI validates the request and enforces the configured prompt-size limit.
2. For chat-completions requests, the firewall extracts text from user-role messages and scores it with the loaded transformer (or the development rule baseline if no model is configured).
3. The policy compares the malicious score with `REVIEW_THRESHOLD` and `BLOCK_THRESHOLD`. Prompts in either restricted band are withheld from the upstream.
4. If the request is allowed and an upstream is configured, the proxy forwards the request, preserving compatible extra request fields and streaming responses.

Implemented endpoints:

- `GET /health`: reports service status, detector mode, model path, and active thresholds.
- `POST /v1/inspect`: returns the predicted class, model scores, policy decision, and inference latency.
- `POST /v1/chat/completions`: inspects user message text, then optionally proxies to an OpenAI-compatible upstream.

The runtime configuration is controlled through environment variables, including `MODEL_PATH`, `BLOCK_THRESHOLD`, `REVIEW_THRESHOLD`, `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, and `UPSTREAM_TIMEOUT_SECONDS`.

## 3. Data and model

The project’s data preparation and training workflow expects CSV columns named `text` and `label`, where `0` means safe and `1` means malicious. The downloader script exports the `core` configuration of `neuralchemy/Prompt-injection-dataset` into train, validation, and test CSV files. The training script fine-tunes a Hugging Face sequence-classification model and can append benign hard-negative examples and balanced replay examples.

The recommended local checkpoint is `artifacts/prompt-firewall-mohamed-fix`. It was fine-tuned from the prior hardened checkpoint with additional benign greeting/introduction examples and replay from both original classes. This follow-up was prompted by a false positive on the harmless message “hi my name is mohamed.” The new checkpoint scored that text at 0.000684 malicious probability during a direct inference check, compared with the previous checkpoint’s score of 0.998116.

Model artifacts and processed data are ignored by Git and are not included in a source-only checkout. Running the trained model therefore requires the local artifact directory to be present.

## 4. Evaluation methodology

The saved validation evaluation selected a blocking threshold to meet a target recall of 90%, choosing the threshold with the lowest false-positive rate among candidates that met the target. That threshold was frozen before evaluating the saved test split. No threshold was selected using the test results.

The selected threshold is **0.9922778606** (typically configured as `0.99228`). The review threshold is **0.75**. Both are fixed policy cutoffs; the model score varies per input. The reported class scores are softmax outputs and have not been calibrated as real-world probabilities of harm.

### Saved evaluation results

| Split | Samples | Accuracy | Precision | Recall | F1 | False-positive rate | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 941 | 94.16% | 99.59% | 90.07% | 94.59% | 0.49% | 405 | 2 | 53 | 481 |
| Test | 942 | 93.10% | 98.60% | 89.49% | 93.83% | 1.79% | 383 | 7 | 58 | 494 |

The validation threshold met its target on the validation split. Test recall was slightly below 90%, with 58 malicious examples missed and 7 benign examples blocked at the blocking threshold. This illustrates why a validation target is not a guarantee of the same result on new traffic.

The test false-positive rate above counts prompts classified as blocked at the blocking threshold. The separate review threshold is lower and causes requests with a score at or above 0.75 to be withheld for review; review-band false-positive rates were not calculated in these saved reports.

### Reproducibility note

The saved metric files (`outputs/validation_metrics_mohamed_fix.json` and `outputs/test_metrics_mohamed_fix.json`) record 941 validation and 942 test examples. At report preparation time, the current files in `data/processed/` have different sizes: 11,256 train, 2,094 validation, and 1,960 test data rows (excluding headers). The current processed CSVs therefore do not match the split sizes recorded in the saved metric artifacts. Treat the metrics above as results from the saved historical evaluation run; preserve or regenerate the exact split files before attempting to reproduce them.

## 5. Results and interpretation

Precision is high on the saved test split: most prompts blocked at the selected threshold were malicious examples from that benchmark. Recall is lower: approximately one in ten malicious examples in the saved test set was missed. The false-positive rate increased from validation to test, from 0.49% to 1.79%, which indicates that benchmark performance varies by split.

The greeting fix improves the demonstrated benign example, but a single direct inference is not a broad evaluation of benign conversation. Synthetic hard negatives can target observed failures but do not replace a larger, diverse and independently labeled benign dataset.

## 6. Deployment and usage

Install the API requirements and start the trained model locally:

```bash
python -m pip install -r requirements-api.txt
MODEL_PATH=artifacts/prompt-firewall-mohamed-fix \
BLOCK_THRESHOLD=0.99228 \
REVIEW_THRESHOLD=0.75 \
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to inspect requests interactively. To proxy to an upstream, set `UPSTREAM_BASE_URL` to its API base and `UPSTREAM_API_KEY` to a valid key. The provided Docker build also needs a reachable registry or a locally cached `python:3.11-slim` image; Docker Hub DNS/network failures happen before the project image is built.

## 7. Limitations and security considerations

- The classifier is probabilistic and can miss attacks or flag benign prompts.
- The training/evaluation benchmark may not represent production prompts, languages, obfuscations, or attacker behavior.
- User-message text is inspected. Image pixels and tool-role messages are not inspected by the current chat route, so multimodal and tool-output prompt injection are outside the implemented inspection coverage.
- The transformer truncates inputs to its maximum sequence length; malicious content beyond that limit may not affect the score.
- The API has no built-in authentication or rate limiting. It should not be exposed publicly without an authenticated gateway and operational controls.
- Scores are uncalibrated softmax values, not probabilities that a prompt is harmful.
- A `review` or `block` decision depends on the configured thresholds. Changing the model checkpoint can invalidate the threshold calibration.
- LLM application security still requires least-privilege tools, server-side authorization, output handling, and human approval for consequential actions.

## 8. Future work

1. Freeze and version the exact dataset split used for each reported model and publish provenance and licensing details.
2. Evaluate on independent prompt-injection benchmarks and a larger benign set, including multilingual and obfuscated examples.
3. Report the operational effect of both the review and blocking thresholds, plus latency distributions under realistic concurrent load.
4. Add inspection support for untrusted tool outputs and text extracted from images/documents where those inputs are accepted.
5. Add API authentication, rate limiting, request-size policy, monitoring, and safe model/version rollout procedures before production use.

## 9. Project files

- `app/`: FastAPI routes, settings, request schemas, and detector implementation.
- `scripts/prepare_data.py`: cleans and stratifies a single labeled CSV.
- `scripts/download_neuralchemy.py`: downloads and exports the Neuralchemy dataset splits.
- `scripts/train.py`: fine-tunes the classifier.
- `scripts/evaluate.py`: evaluates a checkpoint and optionally selects a threshold on validation data.
- `outputs/`: saved JSON evaluation metrics.
- `artifacts/`: local model checkpoints; not tracked by Git.
