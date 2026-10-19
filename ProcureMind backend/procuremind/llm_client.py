"""LLM normalization client for procurement intent extraction.

Reads API credentials from environment variables (supports .env via
python-dotenv) and returns structured JSON-like payload fields.
"""
from __future__ import annotations

import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency for local .env convenience
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


SYSTEM_PROMPT = (
    "You are an extraction engine for product procurement requirements. "
    "Return strict JSON only. No markdown, no prose. "
    "Use only these top-level keys when present: "
    "product_family, max_price, min_ram_gb, min_ssd_gb, max_delivery_days, "
    "category, brand, min_battery_wh, max_weight_kg, min_size_inch, "
    "max_size_inch, min_warranty_years, min_stock_quantity, "
    "min_supplier_reliability_score, max_defect_rate, risk_tolerance, "
    "weights_preset, extras. "
    "If a value is unknown, omit the key. "
    "Put unsupported user details under extras as key-value pairs."
)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output did not contain a JSON object.")
    return json.loads(text[start:end + 1])


def _extract_bedrock_text(response: dict[str, Any]) -> str:
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _bedrock_fallback_models() -> list[str]:
    raw = os.getenv(
        "BEDROCK_FALLBACK_MODELS",
        "anthropic.claude-3-5-sonnet-20241022-v2:0,anthropic.claude-3-haiku-20240307-v1:0",
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


def _is_retired_or_missing_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "resourcenotfoundexception" in text
        or "end of its life" in text
        or "model version has reached the end of its life" in text
        or ("validationexception" in text and "model identifier is invalid" in text)
    )


def _expand_model_id_variants(model_id: str) -> list[str]:
    mid = (model_id or "").strip()
    if not mid:
        return []

    variants = [mid]
    # Some Bedrock deployments require regional prefixes for Anthropic models.
    if mid.startswith("anthropic."):
        variants.append(f"us.{mid}")
    elif mid.startswith("us.anthropic."):
        variants.append(mid.removeprefix("us."))

    out: list[str] = []
    for v in variants:
        if v not in out:
            out.append(v)
    return out


def normalize_with_bedrock(user_text: str,
                           model: str | None = None,
                           timeout_s: int = 12) -> dict[str, Any]:
    """Convert free-form user text into structured payload fields using AWS Bedrock."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ValueError("AWS SDK is not installed. Run: pip install boto3") from exc

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    model_id = (
        model
        or os.getenv("BEDROCK_MODEL_ID")
        or os.getenv("BEDROCK_MODEL")
        or "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # Credentials are resolved by boto3 from env/profile/role, including
    # short-lived session credentials with AWS_SESSION_TOKEN.
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(connect_timeout=timeout_s, read_timeout=timeout_s, retries={"max_attempts": 2}),
    )

    model_candidates: list[str] = []
    for raw_model in [model_id] + [m for m in _bedrock_fallback_models() if m != model_id]:
        for variant in _expand_model_id_variants(raw_model):
            if variant not in model_candidates:
                model_candidates.append(variant)

    last_exc: Exception | None = None
    response: dict[str, Any] | None = None
    for candidate in model_candidates:
        try:
            response = client.converse(
                modelId=candidate,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_text}],
                    }
                ],
                inferenceConfig={"temperature": 0},
            )
            break
        except Exception as exc:
            last_exc = exc
            if _is_retired_or_missing_model_error(exc):
                continue
            raise

    if response is None:
        if last_exc is not None:
            raise last_exc
        raise ValueError("Bedrock call failed with unknown error.")

    text = _extract_bedrock_text(response)
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("LLM output JSON must be an object.")
    return data


def normalize_with_openai(user_text: str,
                          model: str | None = None,
                          timeout_s: int = 12) -> dict[str, Any]:
    """Convert free-form user text into structured payload fields using OpenAI."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ValueError(
            "OpenAI SDK is not installed. Run: pip install openai python-dotenv"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env or environment variables.")

    client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
        timeout=timeout_s,
    )

    text = response.output_text or ""
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("LLM output JSON must be an object.")
    return data


def normalize_with_provider(user_text: str,
                            provider: str = "bedrock",
                            model: str | None = None,
                            timeout_s: int = 12) -> dict[str, Any]:
    provider_name = (provider or "bedrock").strip().lower()
    if provider_name == "bedrock":
        return normalize_with_bedrock(user_text=user_text, model=model, timeout_s=timeout_s)
    if provider_name == "openai":
        return normalize_with_openai(user_text=user_text, model=model, timeout_s=timeout_s)
    raise ValueError(
        f"Unsupported llm_provider '{provider_name}'. Supported: bedrock, openai."
    )