# ProcureMind React Frontend

A React + Vite command center for the uploaded ProcureMind FastAPI backend.

## Backend contract used

The UI calls the existing endpoints:

- `GET /health`
- `POST /api/v1/recommend`
- `POST /api/v1/orchestrate`

The primary workflow uses `/api/v1/orchestrate` with:

```json
{
  "payload": {
    "max_price": 50000,
    "min_ram_gb": 16,
    "min_ssd_gb": 512,
    "max_delivery_days": 7,
    "weights_preset": "ai_heavy",
    "exclude_high_risk": false
  },
  "top_n_offers": 10,
  "use_llm": true,
  "user_text": "I need a laptop...",
  "llm_provider": "bedrock"
}
```

## Run

1. Start the Python/FastAPI backend on port 8000.
2. In this folder:

```bash
npm ci
npm run dev
# Expose frontend on your network (LAN):
npm run dev -- --host 0.0.0.0 --port 5173
```

3. Open the Vite URL shown in the terminal.

Use Node.js 20.19+ (or 22.12+) to match Vite engine requirements.

To use another backend URL, copy `.env.example` to `.env` and change `VITE_API_URL`.

## Important implementation note

The backend exposes recommendation/orchestration and health endpoints, plus persistence endpoints for approval decisions. It does **not** execute purchases, expose a general audit-event endpoint, or provide a dedicated what-if simulator. Therefore:

- Vendor Intelligence uses the actual `ranked_offers` returned by the backend.
- What-If compares actual returned offers client-side; it does not invent a backend calculation.
- Approval Center detects a price-cap exception and records approve/reject/revise decisions through the backend approval endpoints; recording a decision does not execute a purchase.
- Audit Trail combines returned orchestration metadata with browser-session history stored in localStorage. It is not a durable backend audit-event log.

This keeps the frontend honest to the current API contract while still matching the ProcureMind presentation experience.

## UI sections

- Overview dashboard
- Natural-language procurement request
- Live agent pipeline state
- Vendor intelligence / six-factor breakdown
- What-if comparison
- Human approval exception view
- Explainability / audit trail

## Data notice

The backend documentation states that laptop-to-vendor relationships and offer fields are synthetic/derived prototype data. Do not present them as real commercial supplier quotations.
