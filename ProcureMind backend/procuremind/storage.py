from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class ApprovalEvent(Base):
    __tablename__ = "approval_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vendor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vendor_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class VendorDatasetRecord(Base):
    __tablename__ = "vendor_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vendor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vendor_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class FulfillmentLogRecord(Base):
    __tablename__ = "fulfillment_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vendor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    planned_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class PolicyLimitRecord(Base):
    __tablename__ = "policy_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    value_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Storage:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        return self.SessionLocal()

    def save_approval(self,
                      action: str,
                      request_id: str | None,
                      vendor_id: str | None,
                      vendor_name: str | None,
                      reason: str | None,
                      metadata: dict[str, Any] | None) -> dict[str, Any]:
        rec = ApprovalEvent(
            id=f"APR-{uuid4().hex[:10].upper()}",
            action=action,
            request_id=request_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            reason=reason,
            metadata_json=metadata or {},
        )
        with self._session() as s:
            s.add(rec)
            s.commit()
            s.refresh(rec)
        return self._approval_to_dict(rec)

    def list_approvals(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._session() as s:
            rows = s.scalars(
                select(ApprovalEvent)
                .order_by(ApprovalEvent.created_at.desc())
                .limit(limit)
            ).all()
        return [self._approval_to_dict(r) for r in rows]

    def latest_approval(self,
                        request_id: str | None,
                        vendor_id: str | None) -> dict[str, Any] | None:
        stmt = select(ApprovalEvent)
        if request_id is None:
            stmt = stmt.where(ApprovalEvent.request_id.is_(None))
        else:
            stmt = stmt.where(ApprovalEvent.request_id == request_id)
        if vendor_id is None:
            stmt = stmt.where(ApprovalEvent.vendor_id.is_(None))
        else:
            stmt = stmt.where(ApprovalEvent.vendor_id == vendor_id)

        stmt = stmt.order_by(ApprovalEvent.created_at.desc()).limit(1)
        with self._session() as s:
            row = s.scalar(stmt)
        if row is None:
            return None
        return self._approval_to_dict(row)

    def add_vendor_dataset(self,
                           vendor_id: str,
                           vendor_name: str | None,
                           dataset_name: str,
                           payload: dict[str, Any] | None) -> dict[str, Any]:
        rec = VendorDatasetRecord(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            dataset_name=dataset_name,
            payload=payload or {},
        )
        with self._session() as s:
            s.add(rec)
            s.commit()
            s.refresh(rec)
        return self._vendor_dataset_to_dict(rec)

    def list_vendor_datasets(self, vendor_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        stmt = select(VendorDatasetRecord).order_by(VendorDatasetRecord.created_at.desc()).limit(limit)
        if vendor_id:
            stmt = stmt.where(VendorDatasetRecord.vendor_id == vendor_id)
        with self._session() as s:
            rows = s.scalars(stmt).all()
        return [self._vendor_dataset_to_dict(r) for r in rows]

    def add_fulfillment_log(self,
                            vendor_id: str,
                            product_id: str | None,
                            planned_days: float | None,
                            actual_days: float | None,
                            status: str | None,
                            notes: str | None,
                            occurred_at: datetime | None) -> dict[str, Any]:
        rec = FulfillmentLogRecord(
            vendor_id=vendor_id,
            product_id=product_id,
            planned_days=planned_days,
            actual_days=actual_days,
            status=status,
            notes=notes,
            occurred_at=occurred_at or datetime.now(timezone.utc),
        )
        with self._session() as s:
            s.add(rec)
            s.commit()
            s.refresh(rec)
        return self._fulfillment_to_dict(rec)

    def list_fulfillment_logs(self, vendor_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        stmt = select(FulfillmentLogRecord).order_by(FulfillmentLogRecord.occurred_at.desc()).limit(limit)
        if vendor_id:
            stmt = stmt.where(FulfillmentLogRecord.vendor_id == vendor_id)
        with self._session() as s:
            rows = s.scalars(stmt).all()
        return [self._fulfillment_to_dict(r) for r in rows]

    def upsert_policy_limit(self,
                            policy_key: str,
                            value_number: float | None,
                            value_text: str | None,
                            metadata: dict[str, Any] | None) -> dict[str, Any]:
        with self._session() as s:
            rec = s.scalar(select(PolicyLimitRecord).where(PolicyLimitRecord.policy_key == policy_key))
            if rec is None:
                rec = PolicyLimitRecord(
                    policy_key=policy_key,
                    value_number=value_number,
                    value_text=value_text,
                    metadata_json=metadata or {},
                )
                s.add(rec)
            else:
                rec.value_number = value_number
                rec.value_text = value_text
                rec.metadata_json = metadata or {}
                rec.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(rec)
        return self._policy_to_dict(rec)

    def list_policy_limits(self) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(select(PolicyLimitRecord).order_by(PolicyLimitRecord.policy_key.asc())).all()
        return [self._policy_to_dict(r) for r in rows]

    def health(self) -> dict[str, Any]:
        with self._session() as s:
            ts = s.scalar(select(func.now()))
        return {"db": "ok", "time": str(ts)}

    @staticmethod
    def _approval_to_dict(r: ApprovalEvent) -> dict[str, Any]:
        return {
            "id": r.id,
            "timestamp": r.created_at.isoformat(),
            "action": r.action,
            "request_id": r.request_id,
            "vendor_id": r.vendor_id,
            "vendor_name": r.vendor_name,
            "reason": r.reason,
            "metadata": r.metadata_json or {},
        }

    @staticmethod
    def _vendor_dataset_to_dict(r: VendorDatasetRecord) -> dict[str, Any]:
        return {
            "id": r.id,
            "vendor_id": r.vendor_id,
            "vendor_name": r.vendor_name,
            "dataset_name": r.dataset_name,
            "payload": r.payload or {},
            "created_at": r.created_at.isoformat(),
        }

    @staticmethod
    def _fulfillment_to_dict(r: FulfillmentLogRecord) -> dict[str, Any]:
        return {
            "id": r.id,
            "vendor_id": r.vendor_id,
            "product_id": r.product_id,
            "planned_days": r.planned_days,
            "actual_days": r.actual_days,
            "status": r.status,
            "notes": r.notes,
            "occurred_at": r.occurred_at.isoformat(),
        }

    @staticmethod
    def _policy_to_dict(r: PolicyLimitRecord) -> dict[str, Any]:
        return {
            "id": r.id,
            "policy_key": r.policy_key,
            "value_number": r.value_number,
            "value_text": r.value_text,
            "metadata": r.metadata_json or {},
            "updated_at": r.updated_at.isoformat(),
        }


def storage_from_env() -> Storage:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured. Set DATABASE_URL in .env/environment.")
    return Storage(database_url)
