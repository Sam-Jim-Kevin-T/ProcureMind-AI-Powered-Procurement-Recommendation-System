"""ProcureMind backend CLI.

Examples (run inside VS Code terminal):

    # train / retrain the AI models and print CV metrics
    python main.py train

    # end-to-end recommendation (Stage 1 + Stage 2)
    python main.py recommend --max-price 50000 --ram 16 --ssd 512 --delivery 7

    # orchestration mode (flexible payload + deterministic ranking core)
    python main.py orchestrate --payload-file request.json
"""
import argparse
import json

from procuremind import config
from procuremind.orchestration import orchestrate_recommendation
from procuremind.pipeline import ProcureMindEngine
from procuremind.stage1_product_filter import Requirements
from procuremind.models import risk_model


def cmd_train(_args):
    engine = ProcureMindEngine(retrain=True)
    m = engine.train_metrics
    print("\n=== AI MODEL TRAINING (5-fold cross-validation, prototype data) ===")
    print(f"Risk classifier (ExtraTrees) accuracy : {m['risk_classifier'].get('cv_accuracy', float('nan')):.3f}")
    print(f"Risk classifier macro-F1              : {m['risk_classifier'].get('cv_macro_f1', float('nan')):.3f}")
    print(m['risk_classifier'].get('report', ''))
    print(f"Risk regressor  MAE={m['risk_regressor']['cv_mae']:.4f}  R2={m['risk_regressor']['cv_r2']:.3f}")
    print(f"Delay regressor MAE={m['delay_regressor']['cv_mae_days']:.2f} days  R2={m['delay_regressor']['cv_r2']:.3f}")
    print("\nTop features (risk classifier):")
    print(risk_model.feature_importances(engine.profiles, engine.models).head(8).to_string(index=False))
    print("\nNOTE: metrics are prototype metrics on ~100 synthetic/derived "
          "vendor profiles — not production accuracy claims.")


def cmd_recommend(args):
    engine = ProcureMindEngine()
    req = Requirements(max_price=args.max_price, min_ram_gb=args.ram,
                       min_ssd_gb=args.ssd, max_delivery_days=args.delivery,
                       category=args.category, brand=args.brand)
    result = engine.recommend(req, weights_preset=args.weights,
                              exclude_high_risk=args.exclude_high_risk)

    if result["status"] != "ok":
        print(result["message"])
        return

    w = result["weights"]
    print(f"\nWeight preset: '{result['weights_preset']}' — "
          + ", ".join(f"{k.replace('_score','')} {100*v:g}%"
                      for k, v in w.items()))
    if result["exclude_high_risk"]:
        print("High Risk vendors: EXCLUDED "
              f"({result['offers_ranked']} offers remain after exclusion)")
    print(f"\nSTAGE 1: {result['stage1_candidate_products']} candidate laptops, "
          f"{result['stage1_eligible_offers']} eligible offers")
    print("\nTop of shortlist:")
    print(result["shortlist"].head(5).to_string(index=False))

    cols = ["product_id", "vendor_id", "brand", "ram", "ssd_size",
            "offer_price", "delivery_days", "warranty_years", "ai_risk_level",
            "price_score", "specs_score", "delivery_reliability_score",
            "vendor_trust_score", "warranty_score", "price_stability_score",
            "vendor_score", "rank_global"]
    print("\nSTAGE 2: top ranked offers")
    print(result["ranked_offers"][cols].head(10).round(3).to_string(index=False))

    r = result["recommendation"]
    print("\n=== RECOMMENDED ===")
    print(f"Laptop : {r['product_id']} — {r['brand']}, {r['cpu']}, "
          f"{r['ram_gb']:.0f}GB RAM, {r['ssd']} SSD")
    print(f"Vendor : {r['vendor_id']} ({r['vendor_name']})")
    print(f"Price  : {r['offer_price']:.0f} | Delivery: {r['delivery_days']} days "
          f"| Warranty: {r['warranty_years']}y | Risk: {r['ai_risk_level']}")
    print(f"Score  : {r['vendor_score']}/100")
    print(f"Reason : {r['reason']}")
    print(f"\n{result['disclaimer']}")


def cmd_orchestrate(args):
    engine = ProcureMindEngine()

    payload = {}
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if args.payload_json:
        payload.update(json.loads(args.payload_json))

    result = orchestrate_recommendation(engine, payload=payload,
                                        top_n_offers=args.top_n)

    orch = result.get("orchestration", {})
    print("\n=== ORCHESTRATION SUMMARY ===")
    print(f"Mode       : {orch.get('mode', 'n/a')}")
    print(f"Confidence : {orch.get('confidence', 'n/a')}")
    print(f"Agents     : {', '.join(orch.get('agents', []))}")
    if orch.get("warnings"):
        print("Warnings   :")
        for w in orch["warnings"]:
            print(f"  - {w}")

    if result["status"] != "ok":
        print("\nRecommendation status:", result["status"])
        print(result.get("message", "No recommendation available."))
        return

    r = result["recommendation"]
    print("\n=== RECOMMENDED (ORCHESTRATED) ===")
    print(f"Laptop : {r['product_id']} — {r['brand']}, {r['cpu']}, "
          f"{r['ram_gb']:.0f}GB RAM, {r['ssd']} SSD")
    print(f"Vendor : {r['vendor_id']} ({r['vendor_name']})")
    print(f"Price  : {r['offer_price']:.0f} | Delivery: {r['delivery_days']} days "
          f"| Warranty: {r['warranty_years']}y | Risk: {r['ai_risk_level']}")
    print(f"Score  : {r['vendor_score']}/100")
    print(f"Reason : {r['reason']}")
    print(f"\n{result['disclaimer']}")


def build_parser():
    ap = argparse.ArgumentParser(description="ProcureMind two-stage backend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")

    rec = sub.add_parser("recommend")
    rec.add_argument("--max-price", type=float)
    rec.add_argument("--ram", type=float)
    rec.add_argument("--ssd", type=float)
    rec.add_argument("--delivery", type=float)
    rec.add_argument("--category")
    rec.add_argument("--brand")
    rec.add_argument("--weights", choices=sorted(config.WEIGHT_PRESETS),
                     default=None,
                     help="weight preset (default: config.ACTIVE_WEIGHTS = "
                          f"'{config.ACTIVE_WEIGHTS}')")
    rec.add_argument("--exclude-high-risk", action="store_true",
                     help="remove AI-labelled High Risk vendors before ranking")

    orch = sub.add_parser("orchestrate")
    orch.add_argument("--payload-file", default=None,
                      help="path to JSON payload with flexible requirement keys")
    orch.add_argument("--payload-json", default=None,
                      help="inline JSON payload merged with --payload-file")
    orch.add_argument("--top-n", type=int, default=10,
                      help="number of ranked offers to retain")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"train": cmd_train,
     "recommend": cmd_recommend,
     "orchestrate": cmd_orchestrate}[args.cmd](args)
