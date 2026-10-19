# ProcureMind Backend — Two-Stage AI Vendor Recommendation

Python backend implementing the ProcureMind two-stage system with pandas + scikit-learn. Runs directly in VS Code.

## Project structure

```
procuremind_backend/
├── main.py                          # CLI entry point (train / recommend)
├── data/
│   ├── ProcureMind_Relationship_Master.csv   # 25,000 synthetic offers (5,000 laptops × 5 vendors)
│   └── ProcureMind_Vendor_Profiles.csv       # 100 vendor profiles with risk/KPI/SCMS features
├── saved_models/                    # trained .joblib models (created by `train`)
└── procuremind/
    ├── config.py                    # weights (30/20/20/15/10/5), paths, feature lists
    ├── stage1_product_filter.py     # STAGE 1: requirement-based product filtering
    ├── stage2_vendor_ranking.py     # STAGE 2: six-factor scoring + ranking + explanation
    ├── pipeline.py                  # ProcureMindEngine: end-to-end orchestration
    └── models/
        └── risk_model.py            # 3 AI models (see below)
```

## Setup (VS Code / new device)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Required PostgreSQL connection string
# Example:
# DATABASE_URL=postgresql+psycopg://postgres:1916008_pgSQL@localhost:5432/procuremind_db

# Required for orchestration with LLM normalization (default provider: AWS Bedrock)
# set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN / AWS_REGION
# in your environment or .env file.
# Optional: OPENAI_API_KEY can still be used by sending llm_provider=openai.

# Train model artifacts (if missing) and run examples
python main.py train
python main.py recommend --max-price 50000 --ram 16 --ssd 512 --delivery 7
python main.py orchestrate --payload-file request.json
```

### Run API server for the frontend

```bash
uvicorn procuremind.api:app --host 0.0.0.0 --port 8000
```

## PostgreSQL-backed persistence

The API now persists records into PostgreSQL through `DATABASE_URL`.

Stored domains:

- Approval actions (`approve`, `reject`, `revise`)
- Vendor datasets
- Historical fulfillment logs
- Policy limits

Endpoints:

- `POST /api/v1/approvals/action`
- `GET /api/v1/approvals`
- `POST /api/v1/storage/vendor-datasets`
- `GET /api/v1/storage/vendor-datasets`
- `POST /api/v1/storage/fulfillment-logs`
- `GET /api/v1/storage/fulfillment-logs`
- `POST /api/v1/storage/policy-limits`
- `GET /api/v1/storage/policy-limits`

Tables are created automatically on API startup.

## Hybrid agent orchestration (optional)

You can keep deterministic ranking while accepting flexible user data via an
orchestration layer:

- `RequirementExtractionAgent` maps free-form keys (for example `budget`,
  `ram_gb`, `delivery_days`) into the canonical requirement schema.
- `DataQualityAgent` validates numeric inputs against known-data ranges and
  emits confidence + warnings.
- `PolicyAgent` applies lightweight defaults.
- `RankingAgent` calls the existing `ProcureMindEngine.recommend()` pipeline.

Example request payload:

```json
{
  "budget": 50000,
  "ram_gb": 16,
  "ssd": 512,
  "delivery_days": 7,
  "risk_tolerance": "low",
  "preferred_brand": "Dell",
  "new_signal_not_in_schema": 0.92
}
```

Provider notes:

- Default provider is `bedrock`.
- Supported providers are `bedrock` and `openai`.
- For Bedrock, configure AWS credentials (including `AWS_SESSION_TOKEN` when using temporary credentials).
- Optional model override env vars:
  - `BEDROCK_MODEL` (default: `anthropic.claude-3-haiku-20240307-v1:0`)
  - `OPENAI_MODEL` (default: `gpt-4.1-mini`)

Run:

```bash
python main.py orchestrate --payload-file request.json
```

The response includes orchestration metadata (`confidence`, `warnings`,
unknown-field handling) plus the same deterministic recommendation output.

V3 enhancements now included in orchestration:

- Product family auto-detection from user text using weighted taxonomy rules
  (`laptop`, `monitor`, `printer`) with confidence and matched-signal metadata.
- Deterministic extraction of key requirement entities from plain text even
  without LLM usage (budget, delivery, warranty, family-specific constraints).
- `orchestration.product_family_detection` and
  `orchestration.text_extracted_fields` are returned for explainability.

## The three AI models (`procuremind/models/risk_model.py`)

| Model           | Algorithm                 | Target                             | Prototype CV result |
| --------------- | ------------------------- | ---------------------------------- | ------------------- |
| Risk classifier | ExtraTreesClassifier      | risk level (High/Moderate/Low)     | ~0.99 accuracy      |
| Risk regressor  | GradientBoostingRegressor | continuous risk score 0–1          | R² ≈ 0.97           |
| Delay regressor | RandomForestRegressor     | expected delivery deviation (days) | R² ≈ 0.68           |

The classifier keeps the ExtraTrees design from the project summary. The two **regression models** (as requested) make risk continuous and predictive: the risk regressor feeds Vendor Trust (inverse predicted risk) and the delay regressor feeds Delivery Reliability (inverse predicted delay). All models are evaluated with 5-fold cross-validation and persisted with joblib.

**Limitation (keep in the report):** only ~100 vendor profiles exist and the labels are themselves derived from structured/synthetic data, so these are prototype metrics — never claim them as production accuracy.

## Two-stage flow

1. **Stage 1** (`filter_products`): hard constraints — budget, min RAM, min SSD, max delivery days, category, brand — applied product-level and offer-level.
2. **AI predictions** attach `ai_risk_level`, `ai_risk_score`, `ai_expected_delay_days` to every vendor.
3. **Stage 2** (`rank_offers`): official score
   `100 × (0.30·Price + 0.20·Specs + 0.20·DeliveryReliability + 0.15·VendorTrust + 0.10·Warranty + 0.05·PriceStability)`
   with per-component breakdown, global and within-product ranks, and a human-readable reason string.

## Using it programmatically (e.g. from a Flask/FastAPI UI later)

```python
from procuremind.pipeline import ProcureMindEngine
from procuremind.stage1_product_filter import Requirements

engine = ProcureMindEngine()                 # loads data + trained models once
result = engine.recommend(Requirements(
    max_price=50000, min_ram_gb=16, min_ssd_gb=512, max_delivery_days=7))
print(result["recommendation"])
```

`result` also contains the Stage 1 shortlist DataFrame and the full ranked-offers DataFrame with the component breakdown for the "why did this vendor rank first" display.

## Disclaimer

Laptop-to-vendor relationships and offer fields (price, delivery days, warranty, stock) are synthetic/derived for prototype demonstration and must not be presented as real supplier relationships or commercial quotations.
