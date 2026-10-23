import re
import time
import base64
import binascii
import codecs
from dataclasses import dataclass

from app.settings import settings


PATTERNS = [
    re.compile(r"\b(ignore|disregard|override|forget)\b.{0,100}\b(previous|above|prior|system|developer|instructions?)\b", re.I | re.S),
    re.compile(r"\b(reveal|print|repeat|show|leak|exfiltrate)\b.{0,80}\b(system prompt|hidden instructions?|secret|api key|credentials?)\b", re.I | re.S),
    re.compile(r"\b(jailbreak|DAN mode|developer mode|bypass (?:the )?(?:safety|guardrails|filter))\b", re.I),
    re.compile(r"\b(do not follow|don't follow)\b.{0,80}\b(safety|policy|rules|instructions?)\b", re.I | re.S),
]
BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/])")
SEVERITY_CUES = re.compile(r"\b(unrestricted|jailbreak|bypass|exfiltrate|credentials?|api keys?|system prompt|hidden instructions?)\b", re.I)


@dataclass
class Prediction:
    malicious_probability: float
    label: str
    detector: str
    latency_ms: float
    safe_probability: float


class Detector:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.error: str | None = None
        if settings.model_path:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
            self.model.eval()
            self._torch = torch
            labels = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}
            self.malicious_class_id = next(
                (index for index, label in labels.items() if label in {"malicious", "injection", "jailbreak", "label_1"}),
                1,
            )
        except Exception as exc:  # surface a safe readiness error, not model internals
            self.error = f"Could not load configured model ({type(exc).__name__})."
            if not settings.fail_open:
                raise RuntimeError(self.error) from exc

    @property
    def mode(self) -> str:
        return "transformer" if self.model is not None else "rules"

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        probability = self._model_score(text) if self.model is not None else self._rule_score(text)
        # Inspect common text encodings too; attackers may hide overrides from
        # text-only classification with Base64 or ROT13.
        probability = max(probability, self._encoded_injection_score(text), self._rot13_injection_score(text))
        predicted_class = "malicious" if probability >= 0.5 else "safe"
        return Prediction(
            malicious_probability=probability,
            # This is the model's class prediction (argmax), independent from
            # the firewall's configurable block threshold.
            label=predicted_class,
            detector=self.mode,
            latency_ms=(time.perf_counter() - started) * 1000,
            safe_probability=1.0 - probability,
        )

    def _model_score(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
        return float(probabilities[self.malicious_class_id].item())

    @staticmethod
    def _rule_score(text: str) -> float:
        hits = sum(bool(pattern.search(text)) for pattern in PATTERNS)
        if not hits:
            return 0.01
        cue_hits = len(SEVERITY_CUES.findall(text))
        length_bonus = min(len(text) // 160, 3) * 0.005
        return min(0.99, 0.70 + 0.10 * hits + 0.015 * min(cue_hits, 5) + length_bonus)

    @staticmethod
    def _encoded_injection_score(text: str) -> float:
        for match in BASE64_TOKEN.finditer(text):
            token = match.group(0)
            token += "=" * (-len(token) % 4)
            try:
                decoded = base64.b64decode(token, validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            decoded_score = Detector._rule_score(decoded)
            if decoded_score >= 0.70:
                return min(0.999, 0.90 + 0.3 * (decoded_score - 0.70))
        return 0.01

    @staticmethod
    def _rot13_injection_score(text: str) -> float:
        decoded = codecs.decode(text, "rot_13")
        decoded_score = Detector._rule_score(decoded)
        return min(0.999, 0.90 + 0.3 * (decoded_score - 0.70)) if decoded_score >= 0.70 else 0.01


detector = Detector()
