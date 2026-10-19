"""End-to-end ProcureMind pipeline.

    Requirements -> Stage 1 filter -> AI risk predictions ->
    Stage 2 six-factor ranking -> recommendation payload
"""
import pandas as pd

from . import config
from .stage1_product_filter import Requirements, load_master, filter_products, shortlist_products
from .stage2_vendor_ranking import build_vendor_components, rank_offers, explain_offer
from .models import risk_model


class ProcureMindEngine:
    def __init__(self, master_path=None, profiles_path=None, retrain: bool = False):
        self.master = load_master(master_path)
        self.profiles = pd.read_csv(profiles_path or config.VENDOR_PROFILES)

        model_paths = [
            config.RISK_CLASSIFIER_PATH,
            config.RISK_REGRESSOR_PATH,
            config.DELAY_REGRESSOR_PATH,
        ]
        models_exist = all(path.exists() for path in model_paths)

        if retrain or not models_exist:
            self.train_metrics = risk_model.train_all(self.profiles)
            self.models = risk_model.load_models()
        else:
            self.train_metrics = None
            try:
                self.models = risk_model.load_models()
            except Exception:
                # Recover from partial/corrupt artifacts by retraining once.
                self.train_metrics = risk_model.train_all(self.profiles)
                self.models = risk_model.load_models()

        profiles_ai = risk_model.predict_vendor_risk(self.profiles, self.models)
        self.vendor_components = build_vendor_components(profiles_ai)

    # ------------------------------------------------------------ API
    def recommend(self, req: Requirements, top_n_offers: int = 10,
                  weights_preset: str | None = None,
                  exclude_high_risk: bool = False) -> dict:
        """weights_preset: a key of config.WEIGHT_PRESETS (e.g. 'original',
        'vendor_quality'). None = active default (config.ACTIVE_WEIGHTS).
        exclude_high_risk: if True, AI-labelled 'High Risk' vendors are
        removed before Stage 2 ranking."""
        if weights_preset is not None and weights_preset not in config.WEIGHT_PRESETS:
            raise ValueError(f"Unknown weights preset '{weights_preset}'. "
                             f"Available: {list(config.WEIGHT_PRESETS)}")
        weights = (config.WEIGHT_PRESETS[weights_preset]
                   if weights_preset else config.WEIGHTS)
        prioritize_low_risk = (weights_preset == "risk_first")

        eligible = filter_products(self.master, req)          # STAGE 1
        if eligible.empty:
            return {"status": "no_match",
                    "message": "No products satisfy the given requirements.",
                    "requirements": req.as_dict()}

        products = shortlist_products(eligible)
        ranked = rank_offers(eligible, self.vendor_components, req,
                             weights=weights,
                             exclude_high_risk=exclude_high_risk,
                             prioritize_low_risk=prioritize_low_risk)  # STAGE 2
        if ranked.empty:
            return {"status": "no_match",
                    "message": ("All eligible offers come from High Risk "
                                "vendors and were excluded. Re-run without "
                                "--exclude-high-risk or relax requirements."),
                    "requirements": req.as_dict()}

        best = ranked.iloc[0]
        return {
            "status": "ok",
            "product_family": (req.product_family or "laptop"),
            "weights_preset": weights_preset or config.ACTIVE_WEIGHTS,
            "weights": weights,
            "exclude_high_risk": exclude_high_risk,
            "offers_ranked": int(len(ranked)),
            "requirements": req.as_dict(),
            "stage1_candidate_products": int(eligible["product_id"].nunique()),
            "stage1_eligible_offers": int(len(eligible)),
            "shortlist": products,
            "ranked_offers": ranked.head(top_n_offers),
            "recommendation": {
                "product_id": best["product_id"],
                "brand": best["brand"],
                "cpu": best.get("cpu"),
                "ram_gb": (float(best["ram"]) if "ram" in best and pd.notna(best.get("ram")) else None),
                "ssd": best.get("ssd_size"),
                "size_inch": (float(best["size_inch"]) if "size_inch" in best and pd.notna(best.get("size_inch")) else None),
                "vendor_id": best["vendor_id"],
                "vendor_name": best["vendor_name"],
                "offer_price": float(best["offer_price"]),
                "delivery_days": int(best["delivery_days"]),
                "warranty_years": int(best["warranty_years"]),
                "ai_risk_level": best["ai_risk_level"],
                "vendor_score": round(float(best["vendor_score"]), 2),
                "reason": explain_offer(best, weights=weights),
            },
            "disclaimer": ("Laptop-to-vendor links and offer fields are "
                           "synthetic/derived prototype data, not verified "
                           "real-world supplier relationships or quotations."),
        }
