"""Lightweight agent orchestration on top of the deterministic ranking core.

This module keeps Stage 1 + Stage 2 as the source of truth while adding
structured agent steps for flexible user-provided data.
"""
from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np
import pandas as pd

from .llm_client import normalize_with_provider
from .stage1_product_filter import Requirements


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class OrchestrationContext:
    payload: dict[str, Any]
    req: Requirements | None = None
    unknown_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 1.0
    exclude_high_risk: bool = False
    weights_preset: str | None = None
    llm_used: bool = False
    llm_provider: str | None = None
    llm_error: str | None = None
    product_family: str = "laptop"
    family_detection: dict[str, Any] = field(default_factory=dict)
    text_extracted_fields: list[str] = field(default_factory=list)


_FAMILY_ALIASES = {
    "laptop": "laptop",
    "notebook": "laptop",
    "ultrabook": "laptop",
    "pc": "laptop",
    "monitor": "monitor",
    "display": "monitor",
    "screen": "monitor",
    "printer": "printer",
    "laser printer": "printer",
    "inkjet": "printer",
    "laserjet": "printer",
    "generic": "generic",
}

_FAMILY_FIELD_HINTS = {
    "laptop": {"min_ram_gb", "min_ssd_gb", "min_battery_wh", "max_weight_kg"},
    "monitor": {"min_size_inch", "max_size_inch"},
    "printer": {"min_stock_quantity", "min_supplier_reliability_score", "max_defect_rate"},
}

_FAMILY_TAXONOMY = {
    "laptop": [
        (r"\blaptop(s)?\b", 2.0),
        (r"\bnotebook(s)?\b", 1.7),
        (r"\bultrabook(s)?\b", 1.6),
        (r"\bram\b", 0.8),
        (r"\bssd\b", 0.8),
        (r"\bbattery\b", 0.6),
        (r"\bweight\b", 0.4),
    ],
    "monitor": [
        (r"\bmonitor(s)?\b", 2.0),
        (r"\bdisplay(s)?\b", 1.8),
        (r"\bscreen(s)?\b", 1.7),
        (r"\brefresh\s*rate\b", 1.2),
        (r"\bips\b", 0.8),
        (r"\boled\b", 0.8),
        (r"\b(\d{2}(?:\.\d+)?)\s*(?:\"|inches?|inch)\b", 0.8),
    ],
    "printer": [
        (r"\bprinter(s)?\b", 2.0),
        (r"\binkjet\b", 1.7),
        (r"\blaserjet\b", 1.7),
        (r"\blaser\s*printer\b", 1.7),
        (r"\btoner\b", 1.0),
        (r"\bduplex\b", 1.0),
        (r"\bppm\b", 0.9),
    ],
}


def _normalize_family(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[text]
    for alias, canonical in _FAMILY_ALIASES.items():
        if alias in text:
            return canonical
    return None


def _detect_family_from_payload(payload: dict[str, Any]) -> tuple[str | None, dict[str, float], list[str]]:
    if not payload:
        return None, {"laptop": 0.0, "monitor": 0.0, "printer": 0.0}, []

    scores = {"laptop": 0.0, "monitor": 0.0, "printer": 0.0}
    matched: list[str] = []
    for family, keys in _FAMILY_FIELD_HINTS.items():
        for key in keys:
            if key in payload and payload.get(key) not in (None, ""):
                scores[family] += 1.0
                matched.append(key)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if ranked[0][1] <= 0:
        return None, scores, matched
    return ranked[0][0], scores, matched


def _detect_family_from_text(user_text: str | None) -> tuple[str | None, dict[str, float], list[str]]:
    text = (user_text or "").strip().lower()
    scores = {"laptop": 0.0, "monitor": 0.0, "printer": 0.0}
    matched: list[str] = []
    if not text:
        return None, scores, matched

    for family, rules in _FAMILY_TAXONOMY.items():
        for pattern, weight in rules:
            hits = list(re.finditer(pattern, text))
            if not hits:
                continue
            scores[family] += weight * len(hits)
            matched.append(pattern)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if ranked[0][1] <= 0:
        return None, scores, matched
    return ranked[0][0], scores, matched


def _detect_product_family(user_text: str | None, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    direct = _normalize_family((payload or {}).get("product_family"))
    if direct is not None:
        return direct, {
            "source": "payload",
            "confidence": 1.0,
            "scores": {"laptop": 0.0, "monitor": 0.0, "printer": 0.0},
            "matched_signals": ["product_family"],
        }

    payload_family, payload_scores, payload_signals = _detect_family_from_payload(payload or {})
    text_family, text_scores, text_signals = _detect_family_from_text(user_text)

    families = ("laptop", "monitor", "printer")
    combined = {
        fam: payload_scores.get(fam, 0.0) * 1.5 + text_scores.get(fam, 0.0)
        for fam in families
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score <= 0:
        return "generic", {
            "source": "fallback",
            "confidence": 0.3,
            "scores": combined,
            "matched_signals": [],
        }

    margin = best_score - second_score
    confidence = min(0.98, 0.55 + 0.12 * margin)
    source = "combined"
    if payload_family == best and payload_scores.get(best, 0.0) > 0:
        source = "payload_signals"
    elif text_family == best and text_scores.get(best, 0.0) > 0:
        source = "text_taxonomy"

    return best, {
        "source": source,
        "confidence": round(float(confidence), 3),
        "scores": combined,
        "matched_signals": sorted(set(payload_signals + text_signals))[:12],
    }


def _to_score_0_1(value: float) -> float:
    if value > 1:
        return min(1.0, value / 100.0)
    return max(0.0, min(1.0, value))


def _extract_text_requirements(user_text: str | None, family: str) -> dict[str, Any]:
    text = (user_text or "").strip().lower()
    if not text:
        return {}

    out: dict[str, Any] = {}

    money_patterns = [
        r"(?:budget(?:\s*of)?|max(?:imum)?\s*price|price\s*(?:under|below|up\s*to|max(?:imum)?)|cost\s*(?:under|below|up\s*to|max(?:imum)?))\s*(?:is\s*)?(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)",
        r"(?:under|below|up\s*to)\s*(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.\d+)?)",
        r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.\d+)?)",
    ]
    for pattern in money_patterns:
        money = re.search(pattern, text)
        if money:
            out["max_price"] = float(money.group(1).replace(",", ""))
            break

    delivery = re.search(r"(?:within|max(?:imum)?(?:\s*delivery)?|delivery(?:\s*within)?)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:business\s*)?days?\b", text)
    if delivery:
        out["max_delivery_days"] = float(delivery.group(1))

    warranty = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:year|yr)s?\s*warranty\b", text)
    if not warranty:
        warranty = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:year|yr)s?\s*warranty\b", text)
    if warranty:
        out["min_warranty_years"] = float(warranty.group(1))

    category_match = re.search(r"\b(gaming|business|student|ultrabook)\b", text)
    if category_match:
        out["category"] = category_match.group(1).capitalize()

    brand_match = re.search(r"\b(dell|hp|lenovo|asus|acer|apple|msi|samsung|lg|canon|epson|brother)\b", text)
    if brand_match:
        out["brand"] = brand_match.group(1).capitalize()

    if family == "laptop":
        ram = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*gb\s*ram\b|\b([0-9]+(?:\.[0-9]+)?)\s*gb\s*ram\b", text)
        if ram:
            out["min_ram_gb"] = float(ram.group(1) or ram.group(2))

        ssd = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*(tb|gb)\s*(?:ssd|storage)\b|\b([0-9]+(?:\.[0-9]+)?)\s*(tb|gb)\s*ssd\b", text)
        if ssd:
            n = float(ssd.group(1) or ssd.group(3))
            unit = (ssd.group(2) or ssd.group(4) or "gb").lower()
            out["min_ssd_gb"] = n * 1024.0 if unit == "tb" else n

        battery = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*wh\b", text)
        if battery:
            out["min_battery_wh"] = float(battery.group(1))

        weight = re.search(r"(?:under|below|max(?:imum)?|<=?)\s*([0-9]+(?:\.[0-9]+)?)\s*kg\b", text)
        if weight:
            out["max_weight_kg"] = float(weight.group(1))

    elif family == "monitor":
        min_size = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\"|inches?|inch)\b", text)
        if min_size:
            out["min_size_inch"] = float(min_size.group(1))

        max_size = re.search(r"(?:under|below|max(?:imum)?|<=?)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\"|inches?|inch)\b", text)
        if max_size:
            out["max_size_inch"] = float(max_size.group(1))

        if "min_size_inch" not in out and "max_size_inch" not in out:
            plain_size = re.search(r"\b([0-9]{2}(?:\.[0-9]+)?)\s*(?:\"|inches?|inch)\b", text)
            if plain_size:
                out["min_size_inch"] = float(plain_size.group(1))

    elif family == "printer":
        stock = re.search(r"(?:at\s*least|min(?:imum)?|>=?)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units?|stock)\b", text)
        if stock:
            out["min_stock_quantity"] = float(stock.group(1))

        rel_pct = re.search(r"(?:reliability|supplier\s*reliability).{0,20}(?:at\s*least|min(?:imum)?|>=?|above)\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)
        rel_dec = re.search(r"(?:reliability|supplier\s*reliability).{0,20}(?:at\s*least|min(?:imum)?|>=?|above)\s*(0(?:\.[0-9]+)?|1(?:\.0+)?)\b", text)
        if rel_pct:
            out["min_supplier_reliability_score"] = _to_score_0_1(float(rel_pct.group(1)))
        elif rel_dec:
            out["min_supplier_reliability_score"] = _to_score_0_1(float(rel_dec.group(1)))

        defect_pct = re.search(r"(?:defect\s*rate|defects?).{0,24}(?:under|below|max(?:imum)?|<=?)\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)
        defect_dec = re.search(r"(?:defect\s*rate|defects?).{0,24}(?:under|below|max(?:imum)?|<=?)\s*(0(?:\.[0-9]+)?|1(?:\.0+)?)\b", text)
        if defect_pct:
            out["max_defect_rate"] = _to_score_0_1(float(defect_pct.group(1)))
        elif defect_dec:
            out["max_defect_rate"] = _to_score_0_1(float(defect_dec.group(1)))

    if "low risk" in text or "strict risk" in text:
        out["risk_tolerance"] = "low"
    elif "high risk ok" in text or "risk tolerant" in text:
        out["risk_tolerance"] = "high"

    return out


def _friendly_llm_error(err: Exception) -> str:
    """Map provider exceptions to concise user-facing diagnostics."""
    msg = str(err) or err.__class__.__name__
    low = msg.lower()

    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return "LLM normalization unavailable: OpenAI quota is exhausted. Check billing/plan, then retry."
    if "invalid_api_key" in low or "incorrect api key" in low or "authentication" in low:
        return "LLM normalization unavailable: invalid OpenAI API key. Verify OPENAI_API_KEY in backend environment."
    if "model" in low and "not found" in low:
        return "LLM normalization unavailable: configured model is not available for this API key."
    if "timeout" in low:
        return "LLM normalization unavailable: request timed out. Check internet connectivity and retry."
    if "expiredtoken" in low or "token has expired" in low:
        return "LLM normalization unavailable: AWS session token expired. Refresh AWS credentials and retry."
    if "unable to locate credentials" in low or "unrecognizedclientexception" in low:
        return "LLM normalization unavailable: AWS credentials not detected or invalid. Check AWS_* environment variables."
    if "accessdenied" in low:
        return "LLM normalization unavailable: AWS access denied for Bedrock. Verify IAM policy and model access."
    if "validationexception" in low and "modelid" in low:
        return "LLM normalization unavailable: Bedrock model is unavailable in this region/account."

    return f"LLM normalization failed: {msg}"


class RequirementExtractionAgent:
    """Map flexible payload keys into known Requirements fields."""

    KEY_ALIASES = {
        "product_family": ["product_family", "family", "product_type", "item_type"],
        "max_price": ["max_price", "budget", "price_cap"],
        "min_ram_gb": ["min_ram_gb", "ram", "ram_gb", "minimum_ram"],
        "min_ssd_gb": ["min_ssd_gb", "ssd", "ssd_gb", "minimum_ssd"],
        "max_delivery_days": ["max_delivery_days", "delivery", "delivery_days", "max_delivery"],
        "category": ["category", "use_case"],
        "brand": ["brand", "preferred_brand"],
        "min_battery_wh": ["min_battery_wh", "battery_wh", "minimum_battery_wh"],
        "max_weight_kg": ["max_weight_kg", "weight_kg", "max_weight"],
        "min_size_inch": ["min_size_inch", "size_inch_min", "minimum_size_inch", "screen_min_inch"],
        "max_size_inch": ["max_size_inch", "size_inch_max", "maximum_size_inch", "screen_max_inch"],
        "min_warranty_years": ["min_warranty_years", "warranty_years", "minimum_warranty_years"],
        "min_stock_quantity": ["min_stock_quantity", "stock_min", "minimum_stock_quantity", "stock_floor"],
        "min_supplier_reliability_score": [
            "min_supplier_reliability_score",
            "supplier_reliability_min",
            "minimum_supplier_reliability",
            "min_reliability_score",
        ],
        "max_defect_rate": ["max_defect_rate", "defect_rate_max", "maximum_defect_rate"],
    }

    CONTROL_KEYS = {
        "weights", "weights_preset", "exclude_high_risk", "risk_tolerance"
    }

    def run(self, ctx: OrchestrationContext) -> None:
        payload = {str(k).strip(): v for k, v in (ctx.payload or {}).items()}

        mapped: dict[str, Any] = {}
        used_keys: set[str] = set()
        for canonical, aliases in self.KEY_ALIASES.items():
            for key in aliases:
                if key in payload:
                    mapped[canonical] = payload[key]
                    used_keys.add(key)
                    break

        req = Requirements(
            product_family=(str(mapped.get("product_family")).strip().lower() if mapped.get("product_family") is not None else None),
            max_price=_to_float(mapped.get("max_price")),
            min_ram_gb=_to_float(mapped.get("min_ram_gb")),
            min_ssd_gb=_to_float(mapped.get("min_ssd_gb")),
            max_delivery_days=_to_float(mapped.get("max_delivery_days")),
            category=(str(mapped.get("category")).strip() if mapped.get("category") is not None else None),
            brand=(str(mapped.get("brand")).strip() if mapped.get("brand") is not None else None),
            min_battery_wh=_to_float(mapped.get("min_battery_wh")),
            max_weight_kg=_to_float(mapped.get("max_weight_kg")),
            min_size_inch=_to_float(mapped.get("min_size_inch")),
            max_size_inch=_to_float(mapped.get("max_size_inch")),
            min_warranty_years=_to_float(mapped.get("min_warranty_years")),
            min_stock_quantity=_to_float(mapped.get("min_stock_quantity")),
            min_supplier_reliability_score=_to_float(mapped.get("min_supplier_reliability_score")),
            max_defect_rate=_to_float(mapped.get("max_defect_rate")),
        )

        # Normalize empty strings to None for string fields.
        if req.product_family == "":
            req.product_family = None
        if req.category == "":
            req.category = None
        if req.brand == "":
            req.brand = None

        if req.product_family is None:
            req.product_family = ctx.product_family

        ctx.product_family = str(req.product_family or "laptop")

        ctx.req = req

        if "weights_preset" in payload:
            ctx.weights_preset = str(payload["weights_preset"])
            used_keys.add("weights_preset")
        elif "weights" in payload:
            ctx.weights_preset = str(payload["weights"])
            used_keys.add("weights")

        if "exclude_high_risk" in payload:
            ctx.exclude_high_risk = bool(payload["exclude_high_risk"])
            used_keys.add("exclude_high_risk")

        if "risk_tolerance" in payload:
            tol = str(payload["risk_tolerance"]).strip().lower()
            if tol in {"low", "strict"}:
                ctx.exclude_high_risk = True
            used_keys.add("risk_tolerance")

        unknown = [k for k in payload.keys() if k not in used_keys and k not in self.CONTROL_KEYS]
        ctx.unknown_fields = sorted(unknown)


class DataQualityAgent:
    """Validate requirement values against known-data ranges."""

    NUMERIC_FIELDS = {
        "max_price": "offer_price",
        "min_ram_gb": "ram",
        "min_ssd_gb": "ssd_gb",
        "max_delivery_days": "delivery_days",
        "min_battery_wh": "battery_wh",
        "max_weight_kg": "weight_kg",
        "min_size_inch": "size_inch",
        "max_size_inch": "size_inch",
        "min_warranty_years": "warranty_years",
        "min_stock_quantity": "stock_quantity",
        "min_supplier_reliability_score": "supplier_reliability_score",
        "max_defect_rate": "kpi_defect_rate",
    }

    def __init__(self, master: pd.DataFrame):
        self.master = master

    def _warn_if_out_of_distribution(self, req_name: str, value: float, data_col: str, ctx: OrchestrationContext) -> None:
        if data_col not in self.master.columns:
            return
        s = pd.to_numeric(self.master[data_col], errors="coerce")
        s = s.dropna()
        if s.empty:
            return
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        if value < lo or value > hi:
            ctx.warnings.append(
                f"{req_name}={value:g} is outside typical known-data range [{lo:g}, {hi:g}] for {data_col}."
            )
            ctx.confidence -= 0.15

    def run(self, ctx: OrchestrationContext) -> None:
        req = ctx.req
        if req is None:
            ctx.warnings.append("No requirements were extracted from payload.")
            ctx.confidence = 0.0
            return

        req_dict = req.as_dict()
        for req_name, data_col in self.NUMERIC_FIELDS.items():
            if req_name not in req_dict:
                continue
            value = req_dict[req_name]
            if value is None:
                continue
            self._warn_if_out_of_distribution(req_name, float(value), data_col, ctx)

        if ctx.unknown_fields:
            ctx.warnings.append(
                "Unknown payload fields were ignored for scoring: " + ", ".join(ctx.unknown_fields)
            )
            ctx.confidence -= min(0.2, 0.03 * len(ctx.unknown_fields))

        ctx.confidence = float(np.clip(ctx.confidence, 0.0, 1.0))


class PolicyAgent:
    """Apply lightweight policy defaults before ranking."""

    def run(self, ctx: OrchestrationContext) -> None:
        req = ctx.req
        if req is None:
            return

        # If user only supplied weak constraints, keep high-risk offers in pool
        # to avoid accidental no-match outcomes.
        supplied = len(req.as_dict())
        if supplied <= 1:
            ctx.exclude_high_risk = False


class RankingAgent:
    """Delegate ranking to the deterministic pipeline engine."""

    def __init__(self, engine):
        self.engine = engine

    def run(self, ctx: OrchestrationContext, top_n_offers: int = 10) -> dict:
        return self.engine.recommend(
            ctx.req or Requirements(),
            top_n_offers=top_n_offers,
            weights_preset=ctx.weights_preset,
            exclude_high_risk=ctx.exclude_high_risk,
        )


def orchestrate_recommendation(engine,
                               payload: dict[str, Any],
                               top_n_offers: int = 10,
                               user_text: str | None = None,
                               use_llm: bool = False,
                               llm_provider: str = "bedrock") -> dict:
    """Run a small multi-agent workflow and return ranking with orchestration metadata."""
    merged_payload = dict(payload or {})
    ctx = OrchestrationContext(payload=merged_payload)
    family, family_meta = _detect_product_family(user_text, merged_payload)
    ctx.product_family = family
    ctx.family_detection = family_meta

    if family_meta.get("confidence", 0.0) < 0.55:
        ctx.warnings.append(
            "Product family detection confidence is low; defaulting to broad matching behavior."
        )
        ctx.confidence -= 0.08

    extracted = _extract_text_requirements(user_text, ctx.product_family)
    for key, value in extracted.items():
        if key not in merged_payload:
            merged_payload[key] = value
    ctx.text_extracted_fields = sorted(extracted.keys())
    ctx.payload = merged_payload

    if use_llm and user_text:
        provider = (llm_provider or "bedrock").strip().lower()
        ctx.llm_provider = provider
        try:
            llm_payload = normalize_with_provider(user_text, provider=provider)
            if isinstance(llm_payload, dict):
                merged_payload.update(llm_payload)
                ctx.payload = merged_payload
                ctx.llm_used = True
        except Exception as exc:  # fallback to deterministic extraction
            ctx.llm_error = str(exc)
            ctx.warnings.append(_friendly_llm_error(exc))
            ctx.warnings.append("Used deterministic payload mapping as fallback.")
            ctx.confidence -= 0.1

    RequirementExtractionAgent().run(ctx)
    DataQualityAgent(engine.master).run(ctx)
    PolicyAgent().run(ctx)

    result = RankingAgent(engine).run(ctx, top_n_offers=top_n_offers)
    result["orchestration"] = {
        "mode": "hybrid",
        "confidence": round(ctx.confidence, 3),
        "warnings": ctx.warnings,
        "unknown_fields": ctx.unknown_fields,
        "resolved_requirements": (ctx.req.as_dict() if ctx.req else {}),
        "exclude_high_risk_effective": ctx.exclude_high_risk,
        "weights_preset_effective": ctx.weights_preset,
        "llm_used": ctx.llm_used,
        "llm_provider": ctx.llm_provider,
        "llm_error": ctx.llm_error,
        "product_family_effective": ctx.product_family,
        "product_family_detection": ctx.family_detection,
        "text_extracted_fields": ctx.text_extracted_fields,
        "agents": [
            "RequirementExtractionAgent",
            "DataQualityAgent",
            "PolicyAgent",
            "RankingAgent",
        ],
    }
    return result