"""Central configuration for the ProcureMind backend."""
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"

RELATIONSHIP_MASTER = DATA_DIR / "ProcureMind_Relationship_Master.csv"
VENDOR_PROFILES = DATA_DIR / "ProcureMind_Vendor_Profiles.csv"

RISK_CLASSIFIER_PATH = MODEL_DIR / "risk_classifier_extratrees.joblib"
RISK_REGRESSOR_PATH = MODEL_DIR / "risk_regressor.joblib"
DELAY_REGRESSOR_PATH = MODEL_DIR / "delay_regressor.joblib"

# --------------------------------------------- official ranking weights
# Named weight presets (all must sum to 1.0). Selecting a preset here or
# via `--weights <name>` on the CLI covers project step 5 (configurable
# weights).
WEIGHT_PRESETS = {
    # Original model: 30/20/20/15/10/5 (project summary section 6).
    # Price-dominant: a far-cheapest offer can win even from a High Risk
    # vendor (see Test 3).
    "original": {
        "price_score": 0.30,
        "specs_score": 0.20,
        "delivery_reliability_score": 0.20,
        "vendor_trust_score": 0.15,
        "warranty_score": 0.10,
        "price_stability_score": 0.05,
    },
    # Vendor-quality-focused model: 20/20/25/17.5/12.5/5.
    # Price drops from 30% -> 20%; delivery reliability + vendor trust
    # (the AI-model-driven components) rise from 35% -> 42.5% combined,
    # matching the project's focus on vendor qualities.
    "vendor_quality": {
        "price_score": 0.20,
        "specs_score": 0.20,
        "delivery_reliability_score": 0.25,
        "vendor_trust_score": 0.175,
        "warranty_score": 0.125,
        "price_stability_score": 0.05,
    },
    # AI-heavy model: 20/15/25/20/15/5 (recommended default).
    # Specs drops 20% -> 15% because Stage 1 already hard-filters specs
    # (survivors mostly score 1.0, so those points barely discriminate).
    # The freed weight goes to Vendor Trust; the two AI-model-driven
    # components (delivery reliability + vendor trust) now carry 45% of
    # the score, and High Risk offers trail winners by a wider margin on
    # all three official test queries than under "vendor_quality".
    "ai_heavy": {
        "price_score": 0.20,
        "specs_score": 0.15,
        "delivery_reliability_score": 0.25,
        "vendor_trust_score": 0.20,
        "warranty_score": 0.15,
        "price_stability_score": 0.05,
    },
    # Risk-first profile: used with risk-priority sorting mode where
    # lowest ai_risk_score is selected first and weighted score is used
    # as tie-breaker. Keeps prior policies intact while adding an
    # explicit low-risk option.
    "risk_first": {
        "price_score": 0.10,
        "specs_score": 0.10,
        "delivery_reliability_score": 0.15,
        "vendor_trust_score": 0.45,
        "warranty_score": 0.15,
        "price_stability_score": 0.05,
    },
}

# Active preset used by the ranking (change this or pass --weights on CLI)
ACTIVE_WEIGHTS = "ai_heavy"

WEIGHTS = WEIGHT_PRESETS[ACTIVE_WEIGHTS]

# safety check: every preset must sum to exactly 1.0
for _name, _w in WEIGHT_PRESETS.items():
    assert abs(sum(_w.values()) - 1.0) < 1e-9, f"weights '{_name}' do not sum to 1.0"

# ------------------------------------------------- ML feature contract
# Features available in ProcureMind_Vendor_Profiles.csv that describe a
# vendor's operational behaviour (NOT the risk label / score themselves).
RISK_FEATURES = [
    "supplier_reliability_score",
    "lead_time_days_base",
    "shipping_costs_usd_base",
    "warehouse_inventory_level",
    "disruption_likelihood_score",
    "delay_probability",
    "delivery_time_deviation_base",
    "kpi_delivery_success_rate",
    "kpi_compliance_rate",
    "kpi_defect_rate",
    "kpi_avg_unit_price",
    "kpi_avg_negotiated_price",
    "scms_on_time_rate",
    "scms_avg_delivery_deviation_days",
    "scms_avg_unit_price_usd",
    "scms_avg_freight_cost_usd",
]

RISK_LABEL = "risk_level"          # classification target: Low/Moderate/High
RISK_SCORE = "risk_model_score"    # regression target: continuous 0-1 risk
DELAY_TARGET = "delivery_time_deviation_base"  # regression target: delay days

# Risk labels shown in UI are derived from ai_risk_score to keep label and
# percentage consistent. Thresholds can be tuned without code changes.
RISK_LEVEL_THRESHOLDS = {
    "low_max": 0.33,
    "moderate_max": 0.66,
}

assert 0.0 <= RISK_LEVEL_THRESHOLDS["low_max"] < RISK_LEVEL_THRESHOLDS["moderate_max"] <= 1.0, (
    "RISK_LEVEL_THRESHOLDS must satisfy 0 <= low_max < moderate_max <= 1"
)

WARRANTY_MIN_YEARS = 1
WARRANTY_MAX_YEARS = 3
