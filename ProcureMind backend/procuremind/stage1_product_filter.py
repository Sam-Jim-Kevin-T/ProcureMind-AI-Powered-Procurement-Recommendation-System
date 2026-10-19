"""STAGE 1 — Product filtering.

Takes a user requirement object and returns the laptop candidates that
satisfy the hard constraints (budget, RAM, SSD, category, delivery time).
Works on the offer-level relationship master, then collapses to unique
products so Stage 2 can evaluate vendors per shortlisted product.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------- helpers
def storage_to_gb(value) -> float:
    """'512GB' -> 512, '1TB' -> 1024, anything unparsable -> NaN."""
    s = str(value).upper().replace(" ", "")
    import re
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)(TB|GB)", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    return n * 1024 if m.group(2) == "TB" else n


# --------------------------------------------------------- requirements
@dataclass
class Requirements:
    """User purchase requirements (all optional; None = no constraint)."""
    product_family: Optional[str] = None   # laptop, monitor, printer, generic
    max_price: Optional[float] = None      # budget on offer price
    min_ram_gb: Optional[float] = None
    min_ssd_gb: Optional[float] = None
    max_delivery_days: Optional[float] = None
    category: Optional[str] = None         # e.g. 'Gaming', 'Business'
    brand: Optional[str] = None
    min_battery_wh: Optional[float] = None
    max_weight_kg: Optional[float] = None
    min_size_inch: Optional[float] = None
    max_size_inch: Optional[float] = None
    min_warranty_years: Optional[float] = None
    min_stock_quantity: Optional[float] = None
    min_supplier_reliability_score: Optional[float] = None
    max_defect_rate: Optional[float] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ------------------------------------------------------------- stage 1
def load_master(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.RELATIONSHIP_MASTER)
    if "ssd_size" in df.columns:
        df["ssd_gb"] = df["ssd_size"].map(storage_to_gb)
    else:
        df["ssd_gb"] = np.nan
    if "storage" in df.columns:
        df["storage_gb"] = df["storage"].map(storage_to_gb)
    else:
        df["storage_gb"] = np.nan
    return df


def _filter_common(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    """Apply constraints that are valid across product families."""
    if req.category is not None and "category" in df.columns:
        df = df[df["category"].astype(str).str.lower() == req.category.lower()]
    if req.brand is not None and "brand" in df.columns:
        df = df[df["brand"].astype(str).str.lower() == req.brand.lower()]
    if req.max_price is not None and "offer_price" in df.columns:
        df = df[df["offer_price"] <= req.max_price]
    if req.max_delivery_days is not None and "delivery_days" in df.columns:
        df = df[df["delivery_days"] <= req.max_delivery_days]
    if req.min_warranty_years is not None and "warranty_years" in df.columns:
        df = df[df["warranty_years"] >= req.min_warranty_years]
    return df


def _filter_laptop(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    if req.min_ram_gb is not None and "ram" in df.columns:
        df = df[df["ram"] >= req.min_ram_gb]
    if req.min_ssd_gb is not None and "ssd_gb" in df.columns:
        df = df[df["ssd_gb"] >= req.min_ssd_gb]
    if req.min_battery_wh is not None and "battery_wh" in df.columns:
        df = df[df["battery_wh"] >= req.min_battery_wh]
    if req.max_weight_kg is not None and "weight_kg" in df.columns:
        df = df[df["weight_kg"] <= req.max_weight_kg]
    return df


def _filter_monitor(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    if req.min_size_inch is not None and "size_inch" in df.columns:
        df = df[df["size_inch"] >= req.min_size_inch]
    if req.max_size_inch is not None and "size_inch" in df.columns:
        df = df[df["size_inch"] <= req.max_size_inch]
    return df


def _filter_printer(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    if req.min_stock_quantity is not None and "stock_quantity" in df.columns:
        df = df[df["stock_quantity"] >= req.min_stock_quantity]
    if req.min_supplier_reliability_score is not None and "supplier_reliability_score" in df.columns:
        df = df[df["supplier_reliability_score"] >= req.min_supplier_reliability_score]
    if req.max_defect_rate is not None and "kpi_defect_rate" in df.columns:
        df = df[df["kpi_defect_rate"] <= req.max_defect_rate]
    return df


def filter_products(master: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    """Return offer rows whose PRODUCT satisfies the requirements.

    Product-level constraints (RAM, SSD, category, brand...) are applied to
    every offer of that product; offer-level constraints (price, delivery)
    are applied per offer, because different vendors quote different prices
    and delivery times for the same laptop.
    """
    df = master.copy()

    df = _filter_common(df, req)

    family = (req.product_family or "laptop").strip().lower()
    if family == "laptop":
        df = _filter_laptop(df, req)
    elif family == "monitor":
        df = _filter_monitor(df, req)
    elif family == "printer":
        df = _filter_printer(df, req)
    # generic/unknown families use only common constraints

    return df.reset_index(drop=True)


def shortlist_products(filtered_offers: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Collapse eligible offers to one row per product for reporting."""
    if filtered_offers.empty:
        return filtered_offers
    agg_map: dict[str, tuple[str, str]] = {}
    for col in ["brand", "category", "cpu", "ram", "ssd_size", "size_inch"]:
        if col in filtered_offers.columns:
            agg_map[col] = (col, "first")
    if "price" in filtered_offers.columns:
        agg_map["catalog_price"] = ("price", "first")
    if "offer_price" in filtered_offers.columns:
        agg_map["best_offer_price"] = ("offer_price", "min")
    if "delivery_days" in filtered_offers.columns:
        agg_map["fastest_delivery_days"] = ("delivery_days", "min")
    if "vendor_id" in filtered_offers.columns:
        agg_map["eligible_offers"] = ("vendor_id", "nunique")

    agg = filtered_offers.groupby("product_id").agg(**agg_map).reset_index()
    if "best_offer_price" in agg.columns:
        agg = agg.sort_values("best_offer_price")
    return agg.head(top_n)
