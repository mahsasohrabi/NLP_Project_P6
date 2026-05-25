import json
import time
import logging
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)



SYSTEM_PROMPT = """
For each sentence, output ONLY a JSON object with exactly this structure:
{
  "negative": <float between 0 and 1>,
  "neutral": <float between 0 and 1>,
  "positive": <float between 0 and 1>,
  "reasoning": "<one sentence explanation>"
}

The three probabilities must sum to 1.0. Be calibrated: use intermediate values 
(e.g., 0.7/0.2/0.1) when the sentiment is mixed or uncertain. Only use 0.95+ 
when the sentiment is unambiguous."""

USER_TEMPLATE = """Analyze the financial sentiment of this sentence:

"{sentence}"

Respond only with the JSON object."""


# Teacher

class TeacherModel:

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        cache_path: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache: dict = {}
        self._cache_path = Path(cache_path) if cache_path else None

        # Load cache from disk if available
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path) as f:
                self._cache = json.load(f)
            logger.info(f"Loaded {len(self._cache)} cached annotations.")

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    #API 

    def annotate(self, text: str) -> list[float]:
        
        if text in self._cache:
            return self._cache[text]["probs"]

        probs, reasoning = self._call_api(text)
        self._cache[text] = {"probs": probs, "reasoning": reasoning}
        return probs

    def annotate_batch(
        self,
        samples: list,
        save_every: int = 50,
        verbose: bool = True,
    ) -> list:
        
        total = len(samples)
        annotated = 0

        for i, sample in enumerate(samples):
            if sample.soft_labels is not None:
                continue  

            try:
                probs = self.annotate(sample.text)
                sample.soft_labels = probs
                annotated += 1
            except Exception as e:
                logger.warning(f"Failed to annotate sample {i}: {e}")
                one_hot = [0.05, 0.05, 0.05]
                one_hot[sample.label] = 0.85
                sample.soft_labels = one_hot

            if verbose and (i + 1) % 100 == 0:
                print(f"  Annotated {i+1}/{total} samples...")

            # Periodic cache save
            if annotated % save_every == 0 and annotated > 0:
                self._save_cache()

        self._save_cache()
        print(f"✓ Annotation complete. {annotated} new labels generated.")
        return samples

    def estimate_cost(self, n_samples: int) -> dict:
        
        cached = sum(1 for _ in range(min(n_samples, len(self._cache))))
        new = n_samples - cached
        input_tokens  = new * (len(SYSTEM_PROMPT.split()) + 50)
        output_tokens = new * 80
        cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        return {
            "total_samples": n_samples,
            "cached": cached,
            "new_api_calls": new,
            "estimated_cost_usd": round(cost_usd, 4),
            "estimated_time_minutes": round(new * 1.5 / 60, 1),
        }


    def _call_api(self, text: str) -> tuple[list[float], str]:
        """Single API call with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": USER_TEMPLATE.format(sentence=text)
                    }],
                )
                raw = response.content[0].text.strip()
                return self._parse_response(raw)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _parse_response(self, raw: str) -> tuple[list[float], str]:
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse response: {raw[:200]}")

        neg = float(data.get("negative", 0.33))
        neu = float(data.get("neutral", 0.33))
        pos = float(data.get("positive", 0.34))
        reasoning = data.get("reasoning", "")

        total = neg + neu + pos
        if total <= 0:
            total = 1.0
        probs = [neg / total, neu / total, pos / total]

        return probs, reasoning

    def _save_cache(self):
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)


# Groq Teacher (Llama 3.3 70B via OpenAI-compatible API) 

class GroqTeacher:

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    MIN_REQUEST_INTERVAL_SEC = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        retry_delay: float = 4.0,
        cache_path: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache: dict = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._last_call_time = 0.0

        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path) as f:
                self._cache = json.load(f)
            logger.info(f"Loaded {len(self._cache)} cached annotations.")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    # API 

    def annotate(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]["probs"]

        probs, reasoning = self._call_api(text)
        self._cache[text] = {"probs": probs, "reasoning": reasoning}
        return probs

    def annotate_batch(
        self,
        samples: list,
        save_every: int = 25,
        verbose: bool = True,
    ) -> list:
        
        total = len(samples)
        annotated = 0
        quota_exhausted = False

        for i, sample in enumerate(samples):
            if sample.soft_labels is not None:
                continue

            try:
                probs = self.annotate(sample.text)
                sample.soft_labels = probs
                annotated += 1
            except _DailyQuotaExceeded:
                quota_exhausted = True
                logger.warning(
                    f"Daily quota exhausted after {annotated} new annotations. "
                    f"Cache saved. Re-run tomorrow to continue."
                )
                break
            except Exception as e:
                logger.warning(f"Failed to annotate sample {i}: {e}")
                one_hot = [0.05, 0.05, 0.05]
                one_hot[sample.label] = 0.85
                sample.soft_labels = one_hot

            if verbose and (i + 1) % 50 == 0:
                print(f"  Annotated {i+1}/{total} samples (new: {annotated})...")

            if annotated > 0 and annotated % save_every == 0:
                self._save_cache()

        self._save_cache()
        if quota_exhausted:
            remaining = sum(1 for s in samples if s.soft_labels is None)
            print(
                f"⚠ Quota exhausted. {annotated} new labels generated this session. "
                f"{remaining} samples still need annotation — re-run tomorrow."
            )
        else:
            print(f"✓ Annotation complete. {annotated} new labels generated.")
        return samples

    def estimate_cost(self, n_samples: int) -> dict:
        """Free tier — cost is $0. Estimate time only."""
        cached = sum(1 for s_text in [None]*0)  # placeholder
        new = max(0, n_samples - len(self._cache))
        return {
            "total_samples": n_samples,
            "cached": len(self._cache),
            "new_api_calls": new,
            "estimated_cost_usd": 0.0,
            "estimated_time_minutes": round(new * self.MIN_REQUEST_INTERVAL_SEC / 60, 1),
            "daily_quota_warning":
                f"Free tier: 1000 req/day. With {new} new calls needed, "
                f"this will take {(new + 999) // 1000} day(s) of annotation.",
        }


    def _call_api(self, text: str) -> tuple[list[float], str]:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.MIN_REQUEST_INTERVAL_SEC:
            time.sleep(self.MIN_REQUEST_INTERVAL_SEC - elapsed)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": USER_TEMPLATE.format(sentence=text)},
                    ],
                )
                self._last_call_time = time.time()
                raw = response.choices[0].message.content.strip()
                return self._parse_response(raw)

            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str and ("day" in err_str or "rpd" in err_str):
                    raise _DailyQuotaExceeded(str(e))
                if "429" in err_str or "rate" in err_str:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate-limited, waiting {wait:.0f}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(wait)
                    last_error = e
                    continue
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _parse_response(self, raw: str) -> tuple[list[float], str]:
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse response: {raw[:200]}")

        neg = float(data.get("negative", 0.33))
        neu = float(data.get("neutral", 0.33))
        pos = float(data.get("positive", 0.34))
        reasoning = data.get("reasoning", "")

        total = neg + neu + pos
        if total <= 0:
            total = 1.0
        probs = [neg / total, neu / total, pos / total]
        return probs, reasoning

    def _save_cache(self):
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)


class _DailyQuotaExceeded(Exception):
    pass


# Mock Teacher for testing without API

class MockTeacher:

    def __init__(self, confidence: float = 0.80, seed: int = 42):
        import random
        self._rng = random.Random(seed)
        self.confidence = confidence

    def annotate(self, text: str, hard_label: int = 1) -> list[float]:
        noise = (1 - self.confidence) / 2
        probs = [noise, noise, noise]
        probs[hard_label] = self.confidence
        # Add small random jitter
        probs = [p + self._rng.uniform(-0.02, 0.02) for p in probs]
        total = sum(probs)
        return [p / total for p in probs]

    def annotate_batch(self, samples: list, **kwargs) -> list:
        for s in samples:
            s.soft_labels = self.annotate(s.text, s.label)
        return samples
