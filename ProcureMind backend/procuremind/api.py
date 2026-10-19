"""FastAPI interface for frontend integration.

This module exposes HTTP endpoints while keeping the recommendation logic
inside ProcureMindEngine and orchestration helpers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .orchestration import orchestrate_recommendation
from .pipeline import ProcureMindEngine
from .stage1_product_filter import Requirements
from .storage import storage_from_env


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cors_settings() -> tuple[list[str], bool]:
    origins_raw = os.getenv(
        "PROCUREMIND_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    if not origins:
        origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

    allow_credentials = _as_bool(
        os.getenv("PROCUREMIND_ALLOW_CREDENTIALS"), default=False
    )

    # FastAPI/Starlette rejects wildcard origins when credentials are enabled.
    if "*" in origins and allow_credentials:
        allow_credentials = False
    return origins, allow_credentials


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects to JSON-safe Python values."""
    if hasattr(value, "to_dict") and hasattr(value, "empty"):
        # DataFrame-like
        records = value.replace({np.nan: None}).to_dict(orient="records")
        return [{k: _json_safe(v) for k, v in row.items()} for row in records]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is np.nan:
        return None
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _persist_orchestrate_to_storage(req: "OrchestrateRequest", safe_result: dict[str, Any]) -> None:
    """Persist useful orchestration artifacts for operational analytics."""
    storage = app.state.storage

    # Save policy limits derived from incoming payload.
    payload = req.payload or {}
    for key in ("max_price", "max_delivery_days", "exclude_high_risk", "weights_preset"):
        if key in payload:
            val = payload.get(key)
            storage.upsert_policy_limit(
                policy_key=key,
                value_number=_to_float(val),
                value_text=(None if isinstance(val, (int, float, bool)) else str(val)),
                metadata={"source": "orchestrate_request"},
            )

    offers = safe_result.get("ranked_offers") or []
    if not isinstance(offers, list):
        return

    for offer in offers[:50]:
        if not isinstance(offer, dict):
            continue
        vendor_id = str(offer.get("vendor_id") or "").strip()
        if not vendor_id:
            continue

        storage.add_vendor_dataset(
            vendor_id=vendor_id,
            vendor_name=offer.get("vendor_name"),
            dataset_name="ranked_offer_snapshot",
            payload=offer,
        )

        planned = _to_float(offer.get("delivery_days"))
        expected_delay = _to_float(offer.get("ai_expected_delay_days"))
        actual = (planned + expected_delay) if (planned is not None and expected_delay is not None) else None
        storage.add_fulfillment_log(
            vendor_id=vendor_id,
            product_id=(str(offer.get("product_id")) if offer.get("product_id") is not None else None),
            planned_days=planned,
            actual_days=actual,
            status="predicted",
            notes="Auto-generated from orchestration ranking output.",
            occurred_at=None,
        )


class RecommendRequest(BaseModel):
    product_family: str | None = "laptop"
    max_price: float | None = None
    min_ram_gb: float | None = None
    min_ssd_gb: float | None = None
    max_delivery_days: float | None = None
    category: str | None = None
    brand: str | None = None
    min_battery_wh: float | None = None
    max_weight_kg: float | None = None
    min_size_inch: float | None = None
    max_size_inch: float | None = None
    min_warranty_years: float | None = None
    min_stock_quantity: float | None = None
    min_supplier_reliability_score: float | None = None
    max_defect_rate: float | None = None
    top_n_offers: int = Field(default=10, ge=1, le=100)
    weights_preset: str | None = None
    exclude_high_risk: bool = False


class OrchestrateRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    top_n_offers: int = Field(default=10, ge=1, le=100)
    use_llm: bool = False
    user_text: str | None = None
    llm_provider: str = "bedrock"
    product_family: str | None = None


class ApprovalActionRequest(BaseModel):
    action: str
    request_id: str | None = None
    vendor_id: str | None = None
    vendor_name: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VendorDatasetUpsertRequest(BaseModel):
    vendor_id: str
    vendor_name: str | None = None
    dataset_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class FulfillmentLogCreateRequest(BaseModel):
    vendor_id: str
    product_id: str | None = None
    planned_days: float | None = None
    actual_days: float | None = None
    status: str | None = None
    notes: str | None = None
    occurred_at: datetime | None = None


class PolicyLimitUpsertRequest(BaseModel):
    policy_key: str
    value_number: float | None = None
    value_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _app.state.engine = ProcureMindEngine()
    _app.state.storage = storage_from_env()
    _app.state.storage.init_schema()
    yield


app = FastAPI(
    title="ProcureMind API",
    version="1.0.0",
    description="FastAPI endpoints for ProcureMind recommendation and orchestration.",
    lifespan=_lifespan,
)

_allow_origins, _allow_credentials = _cors_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    db = app.state.storage.health()
    return {"status": "ok", "database": db.get("db", "unknown")}


@app.post("/api/v1/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    engine: ProcureMindEngine = app.state.engine
    requirements = Requirements(
        product_family=req.product_family,
        max_price=req.max_price,
        min_ram_gb=req.min_ram_gb,
        min_ssd_gb=req.min_ssd_gb,
        max_delivery_days=req.max_delivery_days,
        category=req.category,
        brand=req.brand,
        min_battery_wh=req.min_battery_wh,
        max_weight_kg=req.max_weight_kg,
        min_size_inch=req.min_size_inch,
        max_size_inch=req.max_size_inch,
        min_warranty_years=req.min_warranty_years,
        min_stock_quantity=req.min_stock_quantity,
        min_supplier_reliability_score=req.min_supplier_reliability_score,
        max_defect_rate=req.max_defect_rate,
    )
    try:
        result = engine.recommend(
            requirements,
            top_n_offers=req.top_n_offers,
            weights_preset=req.weights_preset,
            exclude_high_risk=req.exclude_high_risk,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _json_safe(result)


@app.post("/api/v1/orchestrate")
def orchestrate(req: OrchestrateRequest) -> dict[str, Any]:
    engine: ProcureMindEngine = app.state.engine
    payload = dict(req.payload or {})
    selected_family = str(req.product_family or "").strip().lower()
    if selected_family and selected_family != "auto" and "product_family" not in payload:
        payload["product_family"] = selected_family
    try:
        result = orchestrate_recommendation(
            engine,
            payload=payload,
            top_n_offers=req.top_n_offers,
            user_text=req.user_text,
            use_llm=req.use_llm,
            llm_provider=req.llm_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    safe_result = _json_safe(result)

    # Persist non-critical analytics artifacts without blocking response.
    try:
        _persist_orchestrate_to_storage(req, safe_result)
    except Exception:
        pass

    return safe_result


@app.post("/api/v1/approvals/action")
def approval_action(req: ApprovalActionRequest) -> dict[str, Any]:
    action = (req.action or "").strip().lower()
    allowed = {"approve", "reject", "revise"}
    if action not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Supported: approve, reject, revise.",
        )

    latest = app.state.storage.latest_approval(
        request_id=req.request_id,
        vendor_id=req.vendor_id,
    )
    latest_action = (latest or {}).get("action")
    is_final_locked = latest_action in {"approve", "reject"}
    is_new_final = action in {"approve", "reject"}
    if is_final_locked and is_new_final:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Decision is already finalized as '{latest_action}'. "
                "Submit 'revise' to reopen before another final decision."
            ),
        )

    record = app.state.storage.save_approval(
        action=action,
        request_id=req.request_id,
        vendor_id=req.vendor_id,
        vendor_name=req.vendor_name,
        reason=req.reason,
        metadata=req.metadata,
    )

    return {
        "status": "ok",
        "message": "Approval action saved.",
        "item": record,
    }


@app.get("/api/v1/approvals")
def approvals_list(limit: int = 50) -> dict[str, Any]:
    items = app.state.storage.list_approvals(limit=limit)
    return {"status": "ok", "items": items, "count": len(items)}


@app.post("/api/v1/storage/vendor-datasets")
def vendor_dataset_upsert(req: VendorDatasetUpsertRequest) -> dict[str, Any]:
    item = app.state.storage.add_vendor_dataset(
        vendor_id=req.vendor_id,
        vendor_name=req.vendor_name,
        dataset_name=req.dataset_name,
        payload=req.payload,
    )
    return {"status": "ok", "item": item}


@app.get("/api/v1/storage/vendor-datasets")
def vendor_dataset_list(vendor_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    items = app.state.storage.list_vendor_datasets(vendor_id=vendor_id, limit=limit)
    return {"status": "ok", "items": items, "count": len(items)}


@app.post("/api/v1/storage/fulfillment-logs")
def fulfillment_log_create(req: FulfillmentLogCreateRequest) -> dict[str, Any]:
    item = app.state.storage.add_fulfillment_log(
        vendor_id=req.vendor_id,
        product_id=req.product_id,
        planned_days=req.planned_days,
        actual_days=req.actual_days,
        status=req.status,
        notes=req.notes,
        occurred_at=req.occurred_at,
    )
    return {"status": "ok", "item": item}


@app.get("/api/v1/storage/fulfillment-logs")
def fulfillment_log_list(vendor_id: str | None = None, limit: int = 200) -> dict[str, Any]:
    items = app.state.storage.list_fulfillment_logs(vendor_id=vendor_id, limit=limit)
    return {"status": "ok", "items": items, "count": len(items)}


@app.post("/api/v1/storage/policy-limits")
def policy_limit_upsert(req: PolicyLimitUpsertRequest) -> dict[str, Any]:
    item = app.state.storage.upsert_policy_limit(
        policy_key=req.policy_key,
        value_number=req.value_number,
        value_text=req.value_text,
        metadata=req.metadata,
    )
    return {"status": "ok", "item": item}


@app.get("/api/v1/storage/policy-limits")
def policy_limit_list() -> dict[str, Any]:
    items = app.state.storage.list_policy_limits()
    return {"status": "ok", "items": items, "count": len(items)}