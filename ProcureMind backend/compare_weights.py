"""Compare ranking results under different weight presets.

Runs the three standard test queries under every preset in
config.WEIGHT_PRESETS and prints, per query, the winner side by side —
useful evidence for the report/presentation ("what changed and why").

Usage (VS Code terminal, from the procuremind_backend folder):
    python compare_weights.py
"""
import pandas as pd

from procuremind import config
from procuremind.pipeline import ProcureMindEngine
from procuremind.stage1_product_filter import Requirements

TEST_QUERIES = {
    "TEST 1 (50k, 16GB, 512SSD, 7d)": Requirements(
        max_price=50000, min_ram_gb=16, min_ssd_gb=512, max_delivery_days=7),
    "TEST 2 (80k, 8GB, Gaming)": Requirements(
        max_price=80000, min_ram_gb=8, category="Gaming"),
    "TEST 3 (60k, Dell, 6d)": Requirements(
        max_price=60000, brand="Dell", max_delivery_days=6),
}


def main():
    engine = ProcureMindEngine()
    rows = []

    for test_name, req in TEST_QUERIES.items():
        for preset in config.WEIGHT_PRESETS:
            res = engine.recommend(req, weights_preset=preset)
            if res["status"] != "ok":
                rows.append({"test": test_name, "preset": preset,
                             "winner": "(no match)"})
                continue
            r = res["recommendation"]
            rows.append({
                "test": test_name,
                "preset": preset,
                "laptop": r["product_id"],
                "brand": r["brand"],
                "vendor": r["vendor_id"],
                "price": r["offer_price"],
                "risk": r["ai_risk_level"],
                "score": r["vendor_score"],
            })

    table = pd.DataFrame(rows)
    print("\n=== WINNER PER TEST QUERY, PER WEIGHT PRESET ===\n")
    print(table.to_string(index=False))

    # highlight where the winner actually changed
    print("\n=== DID THE WINNER CHANGE? ===")
    for test_name in TEST_QUERIES:
        sub = table[table["test"] == test_name]
        winners = sub.apply(lambda x: f"{x.get('laptop')}/{x.get('vendor')}",
                            axis=1).unique()
        changed = "YES" if len(winners) > 1 else "no"
        print(f"{test_name}: winner changed = {changed}  ({', '.join(winners)})")

    print("\nNote: synthetic prototype data — not real quotations.")


if __name__ == "__main__":
    main()
