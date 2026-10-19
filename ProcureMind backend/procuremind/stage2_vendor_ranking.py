"""STAGE 2 — Vendor evaluation & ranking.

Implements the six-factor model with configurable weight presets
(config.WEIGHT_PRESETS). Default preset "ai_heavy":
    20% Price + 15% Specs + 25% Delivery Reliability
  + 20% Vendor Trust + 15% Warranty + 5% Price Stability
(Other presets: "vendor_quality" 20/20/25/17.5/12.5/5 and the
original 30/20/20/15/10/5 model as "original".)
Scoring runs on the offers that survived Stage 1, using AI model outputs
(risk classifier + risk regressor + delay regressor) inside the
Delivery Reliability and Vendor Trust components.
"""
import numpy as np
import pandas as pd

from . import config
from .stage1_product_filter import Requirements, storage_to_gb
from .models import risk_model


# --------------------------------------------------------- normalizers
def minmax_high_is_good(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0 or s.max() == s.min():
        return pd.Series(1.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def minmax_low_is_good(s: pd.Series) -> pd.Series:
    return 1.0 - minmax_high_is_good(s)


# ---------------------------------------------------- component scores
def specs_score(offers: pd.DataFrame, req: Requirements) -> pd.Series:
    """How well each laptop fits the requested minimum specs (0-1).

    Identical for vendors selling the same laptop, but differentiates
    laptops from each other in the two-stage recommendation.
    """
    family = (req.product_family or "laptop").strip().lower()
    parts = []

    if family == "laptop":
        if req.min_ram_gb and "ram" in offers.columns:
            parts.append((offers["ram"] / req.min_ram_gb).clip(upper=1.0))
        if req.min_ssd_gb and "ssd_size" in offers.columns:
            ssd = offers["ssd_size"].map(storage_to_gb)
            parts.append((ssd / req.min_ssd_gb).clip(upper=1.0))
        if req.min_battery_wh and "battery_wh" in offers.columns:
            parts.append((offers["battery_wh"] / req.min_battery_wh).clip(upper=1.0))
        if req.max_weight_kg and "weight_kg" in offers.columns:
            parts.append((req.max_weight_kg / offers["weight_kg"]).clip(upper=1.0))

    elif family == "monitor":
        # Monitor fit prioritizes the requested diagonal range.
        if req.min_size_inch and "size_inch" in offers.columns:
            parts.append((offers["size_inch"] / req.min_size_inch).clip(upper=1.0))
        if req.max_size_inch and "size_inch" in offers.columns:
            parts.append((req.max_size_inch / offers["size_inch"]).clip(upper=1.0))

    elif family == "printer":
        # Printer fit emphasizes supply readiness and supplier quality.
        if req.min_stock_quantity and "stock_quantity" in offers.columns:
            parts.append((offers["stock_quantity"] / req.min_stock_quantity).clip(upper=1.0))
        if req.min_supplier_reliability_score and "supplier_reliability_score" in offers.columns:
            parts.append((offers["supplier_reliability_score"] / req.min_supplier_reliability_score).clip(upper=1.0))
        if req.max_defect_rate and "kpi_defect_rate" in offers.columns:
            defect = pd.to_numeric(offers["kpi_defect_rate"], errors="coerce").replace(0, np.nan)
            parts.append((req.max_defect_rate / defect).clip(upper=1.0).fillna(1.0))

    else:
        # V1 generic fallback: rely on common constraints/risk/vendor scoring.
        parts.append(pd.Series(1.0, index=offers.index))

    if not parts:
        return pd.Series(1.0, index=offers.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def build_vendor_components(profiles_with_ai: pd.DataFrame) -> pd.DataFrame:
    """Vendor-level components reused by every offer of that vendor."""
    p = profiles_with_ai.copy()

    # Delivery Reliability: historical on-time behaviour blended with the
    # AI-predicted expected delay (regression model), normalized so that
    # a smaller predicted delay scores higher.
    predicted_delay_score = minmax_low_is_good(p["ai_expected_delay_days"])
    p["delivery_reliability_score"] = (
        0.25 * p["supplier_reliability_score"].clip(0, 1)
        + 0.25 * p["kpi_delivery_success_rate"].clip(0, 1)
        + 0.25 * p["scms_on_time_rate"].clip(0, 1)
        + 0.25 * predicted_delay_score
    )

    # Vendor Trust: reliability + compliance + low defects + inverse
    # AI-predicted risk (regression risk score, continuous).
    p["vendor_trust_score"] = (
        0.35 * p["supplier_reliability_score"].clip(0, 1)
        + 0.25 * p["kpi_compliance_rate"].clip(0, 1)
        + 0.20 * (1 - p["kpi_defect_rate"].clip(0, 1))
        + 0.20 * (1 - p["ai_risk_score"].clip(0, 1))
    )

    # Price Stability: inverse coefficient of variation of historical prices.
    cv_cols = [c for c in ["kpi_price_cv", "kpi_negotiated_price_cv",
                           "scms_price_cv"] if c in p.columns]
    if cv_cols:
        cv = p[cv_cols].mean(axis=1, skipna=True)
    else:
        # derive a proxy CV from available profile price columns
        kpi_cv = (p["kpi_avg_unit_price"].sub(p["kpi_avg_negotiated_price"])
                  .abs() / p["kpi_avg_unit_price"]).clip(0, 2)
        cv = kpi_cv
    p["price_stability_score"] = minmax_low_is_good(cv)

    keep = ["vendor_id", "delivery_reliability_score", "vendor_trust_score",
            "price_stability_score", "ai_risk_level", "ai_risk_score",
            "ai_expected_delay_days"]
    return p[keep]


# --------------------------------------------------------------- rank
def rank_offers(eligible_offers: pd.DataFrame,
                vendor_components: pd.DataFrame,
                req: Requirements,
                weights: dict | None = None,
                exclude_high_risk: bool = False,
                prioritize_low_risk: bool = False) -> pd.DataFrame:
    """Score and rank the offers that survived Stage 1 filtering.

    `weights`: optional weight dict (e.g. a config.WEIGHT_PRESETS entry);
    defaults to the active preset in config.WEIGHTS.
    `exclude_high_risk`: if True, offers from vendors the AI classifier
    labels 'High Risk' are removed BEFORE ranking (hard filter). May
    return an empty frame if every eligible vendor is High Risk.
    `prioritize_low_risk`: if True, ranking is sorted by lowest
    ai_risk_score first; weighted vendor_score is then used as tie-breaker.
    """
    if eligible_offers.empty:
        return eligible_offers

    df = eligible_offers.merge(vendor_components, on="vendor_id", how="left")

    if exclude_high_risk:
        df = df[df["ai_risk_level"] != "High Risk"].reset_index(drop=True)
        if df.empty:
            return df

    df["specs_score"] = specs_score(df, req)
    df["price_score"] = minmax_low_is_good(df["offer_price"])
    df["warranty_score"] = ((df["warranty_years"] - config.WARRANTY_MIN_YEARS)
                            / (config.WARRANTY_MAX_YEARS - config.WARRANTY_MIN_YEARS)
                            ).clip(0, 1)
    for c in ["delivery_reliability_score", "vendor_trust_score",
              "price_stability_score"]:
        df[c] = df[c].clip(0, 1).fillna(0.0)

    w = weights or config.WEIGHTS
    df["vendor_score"] = 100 * (
        w["price_score"] * df["price_score"]
        + w["specs_score"] * df["specs_score"]
        + w["delivery_reliability_score"] * df["delivery_reliability_score"]
        + w["vendor_trust_score"] * df["vendor_trust_score"]
        + w["warranty_score"] * df["warranty_score"]
        + w["price_stability_score"] * df["price_stability_score"]
    )

    if prioritize_low_risk:
        df = df.sort_values(
            ["ai_risk_score", "vendor_score", "price_score"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        df["rank_global"] = np.arange(1, len(df) + 1)
        df["rank_within_product"] = df.groupby("product_id").cumcount() + 1
        return df

    df["rank_global"] = (df["vendor_score"]
                         .rank(method="dense", ascending=False).astype(int))
    df["rank_within_product"] = (df.groupby("product_id")["vendor_score"]
                                 .rank(method="dense", ascending=False)
                                 .astype(int))
    return df.sort_values(["vendor_score", "price_score"],
                          ascending=[False, False]).reset_index(drop=True)


def explain_offer(row: pd.Series, weights: dict | None = None) -> str:
    """Human-readable reason string for the recommendation."""
    w = weights or config.WEIGHTS
    contribs = {
        "price": w["price_score"] * row["price_score"],
        "specs fit": w["specs_score"] * row["specs_score"],
        "delivery reliability": w["delivery_reliability_score"] * row["delivery_reliability_score"],
        "vendor trust": w["vendor_trust_score"] * row["vendor_trust_score"],
        "warranty": w["warranty_score"] * row["warranty_score"],
        "price stability": w["price_stability_score"] * row["price_stability_score"],
    }
    top = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    strongest = ", ".join(f"{name} ({100*val:.1f} pts)" for name, val in top)
    return (f"Score {row['vendor_score']:.2f}/100. Strongest components: "
            f"{strongest}. AI risk: {row['ai_risk_level']} "
            f"(score {row['ai_risk_score']:.2f}), predicted delivery "
            f"deviation ≈ {row['ai_expected_delay_days']:.1f} days. "
            f"[Synthetic prototype offer — not a real quotation.]")
