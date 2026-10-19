import React, { useEffect, useMemo, useState } from 'react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const STORAGE_KEY = 'procuremind.history.v1';

const AGENTS = [
  ['RequirementExtractionAgent', 'Understands the procurement brief'],
  ['DataQualityAgent', 'Checks data ranges and confidence'],
  ['PolicyAgent', 'Applies risk and policy defaults'],
  ['RankingAgent', 'Runs deterministic two-stage ranking'],
];

const NAV = [
  ['dashboard', 'Overview', '⌂'],
  ['request', 'New Request', '＋'],
  ['vendors', 'Vendor Intelligence', '◈'],
  ['whatif', 'What-If', '◇'],
  ['approvals', 'Approvals', '✓'],
  ['audit', 'Audit Trail', '≡'],
];

const demoRequest = {
  product_family: 'auto',
  max_price: 50000,
  min_ram_gb: 16,
  min_ssd_gb: 512,
  max_delivery_days: 7,
  category: '',
  brand: '',
  min_battery_wh: '',
  max_weight_kg: '',
  min_size_inch: '',
  max_size_inch: '',
  min_warranty_years: '',
  min_stock_quantity: '',
  min_supplier_reliability_score: '',
  max_defect_rate: '',
  top_n_offers: 10,
  weights_preset: 'ai_heavy',
  exclude_high_risk: false,
};

function money(value) {
  if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) return '—';
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(value));
}
function pct(value) {
  const n = Number(value);
  if (Number.isNaN(n)) return '—';
  return `${Math.round(n <= 1 ? n * 100 : n)}%`;
}
function score(value) {
  const n = Number(value);
  return Number.isNaN(n) ? '—' : n.toFixed(1);
}
function riskClass(risk) {
  const r = String(risk || '').toLowerCase();
  return r.includes('high') ? 'danger' : r.includes('moderate') ? 'warn' : 'safe';
}
function cleanPayload(form) {
  const p = {};
  const numeric = [
    'max_price','min_ram_gb','min_ssd_gb','max_delivery_days','min_battery_wh','max_weight_kg',
    'min_size_inch','max_size_inch','min_warranty_years','min_stock_quantity',
    'min_supplier_reliability_score','max_defect_rate'
  ];
  numeric.forEach(k => { if (form[k] !== '' && form[k] !== null && form[k] !== undefined) p[k] = Number(form[k]); });
  if (form.product_family && form.product_family !== 'auto') p.product_family = form.product_family;
  ['category','brand'].forEach(k => { if (form[k]) p[k] = form[k]; });
  if (form.weights_preset) p.weights_preset = form.weights_preset;
  p.exclude_high_risk = Boolean(form.exclude_high_risk);
  return p;
}

async function api(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      // Required for free ngrok URLs to bypass interstitial warning page for API calls.
      'ngrok-skip-browser-warning': '1',
      ...(options.headers || {}),
    },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.detail || data?.message || `Request failed (${res.status})`);
  return data;
}

function Icon({ children }) { return <span className="nav-icon" aria-hidden="true">{children}</span>; }

function App() {
  const [page, setPage] = useState('dashboard');
  const [form, setForm] = useState(demoRequest);
  const [brief, setBrief] = useState('');
  const [useLLM, setUseLLM] = useState(true);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState('checking');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; }
  });
  const [approvalEvents, setApprovalEvents] = useState([]);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [selectedOffer, setSelectedOffer] = useState(null);
  const [toast, setToast] = useState('');

  useEffect(() => {
    api('/health').then(() => setHealth('online')).catch(() => setHealth('offline'));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 20)));
  }, [history]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(''), 2800);
    return () => clearTimeout(t);
  }, [toast]);

  const offers = result?.ranked_offers || [];
  const rec = result?.recommendation;
  const orch = result?.orchestration || {};

  useEffect(() => {
    api('/api/v1/approvals')
      .then(data => setApprovalEvents(Array.isArray(data?.items) ? data.items : []))
      .catch(() => setApprovalEvents([]));
  }, []);

  const metrics = useMemo(() => {
    const risks = offers.reduce((a, x) => a + (String(x.ai_risk_level || '').toLowerCase() === 'high risk' ? 1 : 0), 0);
    const avg = offers.length ? offers.reduce((a, x) => a + Number(x.vendor_score || 0), 0) / offers.length : 0;
    return {
      requests: history.length,
      offers: result?.offers_ranked || 0,
      highRisk: risks,
      avgScore: avg,
    };
  }, [offers, result, history]);

  async function runProcurement() {
    setError('');
    setLoading(true);
    try {
      const payload = cleanPayload(form);
      const data = await api('/api/v1/orchestrate', {
        method: 'POST',
        body: JSON.stringify({
          payload,
          product_family: (form.product_family && form.product_family !== 'auto') ? form.product_family : null,
          top_n_offers: Number(form.top_n_offers || 10),
          use_llm: Boolean(useLLM && brief.trim()),
          user_text: brief.trim() || null,
          llm_provider: 'bedrock',
        }),
      });
      setResult(data);
      setSelectedOffer(data?.ranked_offers?.[0] || null);
      const entry = {
        id: `PM-${Date.now().toString(36).toUpperCase()}`,
        time: new Date().toISOString(),
        status: data.status,
        recommendation: data.recommendation || null,
        confidence: data.orchestration?.confidence ?? null,
        requirements: data.orchestration?.resolved_requirements || data.requirements || payload,
      };
      setHistory(prev => [entry, ...prev].slice(0, 20));
      setPage('dashboard');
      setToast(data.status === 'ok' ? 'Procurement analysis completed' : 'No matching offer found');
    } catch (e) {
      setError(e.message);
      setToast('Could not reach ProcureMind API');
    } finally { setLoading(false); }
  }

  function openHistory(item) {
    if (!item.recommendation) return;
    setResult({
      status: item.status,
      recommendation: item.recommendation,
      requirements: item.requirements,
      orchestration: { confidence: item.confidence, mode: 'history', agents: AGENTS.map(a => a[0]), warnings: [] },
      ranked_offers: [],
      offers_ranked: 0,
      disclaimer: 'Historical result shown from browser session storage.',
    });
    setPage('dashboard');
  }

  async function submitApprovalAction(action) {
    if (!result?.recommendation) {
      setToast('Run procurement before submitting an approval action');
      return;
    }

    setApprovalBusy(true);
    try {
      const r = result.recommendation;
      const req = result.requirements || result.orchestration?.resolved_requirements || {};
      const requestId = history?.[0]?.id || null;
      const body = {
        action,
        request_id: requestId,
        vendor_id: r.vendor_id || null,
        vendor_name: r.vendor_name || null,
        reason: action === 'revise' ? 'Requested revision from Approval Center.' : null,
        metadata: {
          offer_price: r.offer_price,
          max_price: req.max_price ?? null,
          ai_risk_level: r.ai_risk_level ?? null,
          vendor_score: r.vendor_score ?? null,
        },
      };

      await api('/api/v1/approvals/action', {
        method: 'POST',
        body: JSON.stringify(body),
      });

      const latest = await api('/api/v1/approvals');
      setApprovalEvents(Array.isArray(latest?.items) ? latest.items : []);
      setToast(`Approval action saved: ${action}`);
    } catch (e) {
      setToast(e.message || 'Could not save approval action');
    } finally {
      setApprovalBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand" onClick={() => setPage('dashboard')}>
          <div className="brand-mark"><span></span><span></span><span></span></div>
          <div><strong>PROCURE<span>MIND</span></strong><small>AI PROCUREMENT OS</small></div>
        </div>
        <div className="nav-label">COMMAND CENTER</div>
        <nav>
          {NAV.map(([id, label, icon]) => (
            <button key={id} className={`nav-item ${page === id ? 'active' : ''}`} onClick={() => setPage(id)}>
              <Icon>{icon}</Icon><span>{label}</span>
              {id === 'approvals' && pendingApproval(result) ? <b className="nav-badge">1</b> : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-card">
            <div className={`status-dot ${health}`}></div>
            <div><strong>Backend</strong><small>{health === 'online' ? 'FastAPI connected' : health === 'checking' ? 'Checking API…' : 'Offline / unavailable'}</small></div>
          </div>
          <div className="api-url">{API_URL}</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div><div className="eyebrow">PROCUREMENT COMMAND CENTER</div><h1>{pageTitle(page)}</h1></div>
          <div className="top-actions">
            <div className="live-pill"><span className="pulse"></span> AI ENGINE {health === 'online' ? 'LIVE' : 'STANDBY'}</div>
            <button className="avatar">ES</button>
          </div>
        </header>

        <div className="content">
          {page === 'dashboard' && <Dashboard result={result} metrics={metrics} history={history} onNew={() => setPage('request')} onHistory={openHistory} onViewAll={() => setPage('vendors')} />}
          {page === 'request' && <RequestPage form={form} setForm={setForm} brief={brief} setBrief={setBrief} useLLM={useLLM} setUseLLM={setUseLLM} onRun={runProcurement} loading={loading} error={error} />}
          {page === 'vendors' && <VendorsPage offers={offers} recommendation={rec} selected={selectedOffer} setSelected={setSelectedOffer} />}
          {page === 'whatif' && <WhatIfPage offers={offers} recommendation={rec} />}
          {page === 'approvals' && <ApprovalsPage result={result} approvalEvents={approvalEvents} onAction={submitApprovalAction} busy={approvalBusy} currentRequestId={history?.[0]?.id || null} />}
          {page === 'audit' && <AuditPage result={result} history={history} />}
        </div>
      </main>
      {toast && <div className="toast">✓ {toast}</div>}
    </div>
  );
}

function pageTitle(page) {
  return { dashboard: 'Overview', request: 'Create Procurement Request', vendors: 'Vendor Intelligence', whatif: 'What-If Simulator', approvals: 'Approval Center', audit: 'Audit Trail' }[page] || 'ProcureMind';
}
function pendingApproval(result) {
  const r = result?.recommendation, req = result?.requirements || result?.orchestration?.resolved_requirements;
  return Boolean(r && req?.max_price && Number(r.offer_price) > Number(req.max_price));
}

function noMatchTips(requirements = {}) {
  const tips = [];
  if (requirements.category) tips.push('Try setting Category to Any category.');
  if (requirements.brand) tips.push('Try removing Preferred brand.');
  if (requirements.max_price !== undefined) tips.push('Try increasing Maximum price by 10-20%.');
  if (requirements.max_delivery_days !== undefined) tips.push('Try increasing Max delivery days.');
  if (requirements.exclude_high_risk) tips.push('Try turning off Exclude High Risk vendors.');
  if (!tips.length) tips.push('Try relaxing one or two filters and run again.');
  return tips.slice(0, 4);
}

function Dashboard({ result, metrics, history, onNew, onHistory, onViewAll }) {
  const r = result?.recommendation;
  const weights = result?.weights || {};
  const top = result?.ranked_offers || [];
  return <>
    <section className="hero-grid">
      <div className="hero-card">
        <div className="hero-orb"></div>
        <div className="hero-copy">
          <div className="tag">AUTONOMOUS PROCUREMENT</div>
          <h2>Turn a buying brief into an <em>explainable decision.</em></h2>
          <p>ProcureMind combines intent extraction, data quality, policy, AI risk signals and deterministic vendor ranking in one workflow.</p>
          <button className="primary-btn" onClick={onNew}>＋ Start procurement</button>
        </div>
        <div className="hero-flow">
          {['BRIEF','EXTRACT','FILTER','RANK','DECIDE'].map((x,i) => <React.Fragment key={x}><span>{x}</span>{i<4 && <b>→</b>}</React.Fragment>)}
        </div>
      </div>
      <div className="score-card">
        <div className="card-top"><span>RECOMMENDATION</span><span className="mini-dot"></span></div>
        {r ? <>
          <div className="score-ring"><strong>{score(r.vendor_score)}</strong><small>/100</small></div>
          <h3>{r.vendor_name || r.vendor_id}</h3>
          <p>{r.brand} · {r.cpu} · {r.ram_gb}GB · {r.ssd}</p>
          <div className="recommend-row"><span>{money(r.offer_price)}</span><span>{r.delivery_days}d delivery</span><span className={`risk ${riskClass(r.ai_risk_level)}`}>{r.ai_risk_level}</span></div>
        </> : <div className="empty-state compact"><div className="empty-icon">◎</div><strong>No decision yet</strong><p>Run a procurement request to populate the command center.</p></div>}
      </div>
    </section>

    <section className="metric-grid">
      <Metric label="REQUESTS ANALYZED" value={metrics.requests} icon="◌" detail="This browser session" />
      <Metric label="ELIGIBLE OFFERS" value={metrics.offers} icon="◈" detail="Latest ranked pool" />
      <Metric label="HIGH-RISK OFFERS" value={metrics.highRisk} icon="△" detail="In latest shortlist" tone="danger" />
      <Metric label="AVG. VENDOR SCORE" value={metrics.avgScore ? metrics.avgScore.toFixed(1) : '—'} icon="✦" detail="Latest ranked offers" tone="accent" />
    </section>

    {result?.status === 'no_match' ? <section className="panel"><div className="panel-head"><div><h3>NO MATCH FOUND</h3><p>{result?.message || 'No offers matched the current constraints.'}</p></div></div><div className="warning-box">⚠ Suggested fixes: {noMatchTips(result?.requirements || result?.orchestration?.resolved_requirements).join(' ')}</div></section> : null}

    <section className="two-col">
      <Panel title="AI DECISION PIPELINE" subtitle="Latest orchestration state">
        <AgentTimeline orchestration={result?.orchestration} hasResult={Boolean(result)} />
      </Panel>
      <Panel title="SCORING MODEL" subtitle={result?.weights_preset ? `Preset · ${result.weights_preset}` : 'AI-heavy default'}>
        <WeightBars weights={weights} />
      </Panel>
    </section>

    <section className="two-col lower">
      <Panel title="LATEST RANKED OFFERS" subtitle={top.length ? `${top.length} offers returned by Stage 2` : 'Run a request to see vendor intelligence'} action={top.length ? <button className="text-btn" onClick={onViewAll}>View all →</button> : null}>
        {top.length ? <MiniOffers offers={top.slice(0,5)} /> : <EmptyPanel text="Your top vendors, risk labels and six-factor scores will appear here." />}
      </Panel>
      <Panel title="RECENT REQUESTS" subtitle="Local session history">
        {history.length ? history.slice(0,5).map(h => <button className="history-row" key={h.id} onClick={() => onHistory(h)}><span className="history-id">{h.id}</span><span>{h.recommendation?.vendor_name || 'No match'}</span><span className={`status ${h.status === 'ok' ? 'safe' : 'warn'}`}>{h.status}</span></button>) : <EmptyPanel text="No requests have been analyzed yet." />}
      </Panel>
    </section>
    <div className="prototype-note">Prototype data notice: vendor relationships and offers are synthetic/derived and should not be treated as real supplier quotations.</div>
  </>;
}

function Metric({label,value,icon,detail,tone=''}) { return <div className={`metric ${tone}`}><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></div>; }
function Panel({title,subtitle,children,action}) { return <section className="panel"><div className="panel-head"><div><h3>{title}</h3><p>{subtitle}</p></div>{action}</div>{children}</section>; }
function EmptyPanel({text}) { return <div className="empty-panel">{text}</div>; }

function AgentTimeline({orchestration, hasResult}) {
  const completed = hasResult;
  return <div className="agent-timeline">{AGENTS.map(([name,desc],i) => <div className="agent-step" key={name}><div className={`agent-node ${completed ? 'done' : ''}`}>{completed ? '✓' : String(i+1).padStart(2,'0')}</div><div className="agent-info"><strong>{name.replace('Agent','')}</strong><span>{desc}</span></div><span className={`agent-state ${completed ? 'done' : ''}`}>{completed ? 'COMPLETED' : 'WAITING'}</span></div>)}{orchestration?.warnings?.length ? <div className="warning-box">⚠ {orchestration.warnings.join(' ')}</div> : null}</div>;
}
function WeightBars({weights}) {
  const labels = {price_score:'Price', specs_score:'Specs', delivery_reliability_score:'Delivery reliability', vendor_trust_score:'Vendor trust', warranty_score:'Warranty', price_stability_score:'Price stability'};
  const entries = Object.keys(labels).map(k => [labels[k], Number(weights[k] || 0)]);
  if (!entries.some(([,v]) => v)) return <div className="weight-placeholder"><strong>20 / 15 / 25 / 20 / 15 / 5</strong><span>Price · Specs · Delivery · Trust · Warranty · Stability</span></div>;
  return <div className="weights">{entries.map(([label,v]) => <div className="weight" key={label}><div><span>{label}</span><b>{Math.round(v*100)}%</b></div><div className="bar"><i style={{width:`${v*100}%`}} /></div></div>)}</div>;
}
function MiniOffers({offers}) { return <div className="mini-offers">{offers.map((o,i) => <div className="mini-offer" key={`${o.vendor_id}-${o.product_id}-${i}`}><span className="rank">#{i+1}</span><div className="vendor-avatar">{String(o.vendor_name || o.vendor_id || 'V').slice(0,1)}</div><div className="mini-main"><strong>{o.vendor_name || o.vendor_id}</strong><span>{o.brand} {o.product_id} · {o.delivery_days}d</span></div><div className="mini-price">{money(o.offer_price)}</div><div className={`score-chip ${riskClass(o.ai_risk_level)}`}>{score(o.vendor_score)}</div></div>)}</div>; }

function RequestPage({form,setForm,brief,setBrief,useLLM,setUseLLM,onRun,loading,error}) {
  const set = (k,v) => setForm(prev => ({...prev,[k]:v}));
  const family = form.product_family || 'laptop';
  return <div className="request-layout">
    <section className="request-main">
      <div className="section-intro"><div className="tag">NATURAL LANGUAGE INTAKE</div><h2>What do you need to buy?</h2><p>Describe your procurement need in plain language. The optional LLM layer converts the brief into the backend's canonical requirement schema before deterministic scoring.</p></div>
      <textarea className="brief-box" value={brief} onChange={e=>setBrief(e.target.value)} placeholder="Example: I need a Dell laptop under ₹60,000 with at least 16GB RAM and 512GB SSD, delivered within 7 days. Prefer low-risk vendors." />
      <div className="llm-row"><label className="switch"><input type="checkbox" checked={useLLM} onChange={e=>setUseLLM(e.target.checked)} /><span></span></label><div><strong>Use LLM normalization</strong><small>Provider-based extraction is used only when a brief is provided; deterministic mapping remains the fallback.</small></div><span className="llm-badge">LLM</span></div>
      <div className="divider-label"><span>OR CONFIGURE CONSTRAINTS</span></div>
      <div className="form-grid">
        <Field label="Product family"><select value={form.product_family} onChange={e=>set('product_family',e.target.value)}><option value="auto">Auto detect from brief</option><option value="laptop">Laptop</option><option value="monitor">Monitor</option><option value="printer">Printer</option><option value="generic">Generic</option></select></Field>
        <Field label="Maximum price (₹)"><input type="number" value={form.max_price} onChange={e=>set('max_price',e.target.value)} /></Field>
        <Field label="Max delivery (days)"><input type="number" value={form.max_delivery_days} onChange={e=>set('max_delivery_days',e.target.value)} /></Field>
        <Field label="Category"><select value={form.category} onChange={e=>set('category',e.target.value)}><option value="">Any category</option><option>Gaming</option><option>Business</option><option>Student</option><option>Ultrabook</option></select></Field>
        <Field label="Preferred brand"><input value={form.brand} onChange={e=>set('brand',e.target.value)} placeholder="Any brand" /></Field>
        <Field label="Minimum warranty (years)"><input type="number" value={form.min_warranty_years} onChange={e=>set('min_warranty_years',e.target.value)} placeholder="Optional" /></Field>
        {family === 'laptop' ? <Field label="Minimum RAM (GB)"><input type="number" value={form.min_ram_gb} onChange={e=>set('min_ram_gb',e.target.value)} /></Field> : null}
        {family === 'laptop' ? <Field label="Minimum SSD (GB)"><input type="number" value={form.min_ssd_gb} onChange={e=>set('min_ssd_gb',e.target.value)} /></Field> : null}
        {family === 'laptop' ? <Field label="Minimum battery (Wh)"><input type="number" value={form.min_battery_wh} onChange={e=>set('min_battery_wh',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'laptop' ? <Field label="Maximum weight (kg)"><input type="number" value={form.max_weight_kg} onChange={e=>set('max_weight_kg',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'monitor' ? <Field label="Minimum size (inch)"><input type="number" value={form.min_size_inch} onChange={e=>set('min_size_inch',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'monitor' ? <Field label="Maximum size (inch)"><input type="number" value={form.max_size_inch} onChange={e=>set('max_size_inch',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'printer' ? <Field label="Minimum stock quantity"><input type="number" value={form.min_stock_quantity} onChange={e=>set('min_stock_quantity',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'printer' ? <Field label="Minimum supplier reliability (0-1)"><input type="number" min="0" max="1" step="0.01" value={form.min_supplier_reliability_score} onChange={e=>set('min_supplier_reliability_score',e.target.value)} placeholder="Optional" /></Field> : null}
        {family === 'printer' ? <Field label="Maximum defect rate (0-1)"><input type="number" min="0" max="1" step="0.01" value={form.max_defect_rate} onChange={e=>set('max_defect_rate',e.target.value)} placeholder="Optional" /></Field> : null}
      </div>
      <div className="advanced-row"><div><strong>Ranking policy</strong><small>Choose how the six-factor scoring should prioritize offers.</small></div><select value={form.weights_preset} onChange={e=>set('weights_preset',e.target.value)}><option value="ai_heavy">AI Heavy · recommended</option><option value="risk_first">Risk First · lowest risk priority</option><option value="vendor_quality">Vendor Quality</option><option value="original">Original · price dominant</option></select></div>
      <div className="advanced-row"><div><strong>Exclude High Risk vendors</strong><small>Hard-filter AI-labelled High Risk offers before ranking.</small></div><label className="switch"><input type="checkbox" checked={form.exclude_high_risk} onChange={e=>set('exclude_high_risk',e.target.checked)} /><span></span></label></div>
      {error && <div className="error-box">× {error}</div>}
      <button className="primary-btn run-btn" onClick={onRun} disabled={loading}>{loading ? <><span className="spinner"></span> Running agents…</> : <>Run ProcureMind <span>→</span></>}</button>
    </section>
    <aside className="request-side">
      <Panel title="WORKFLOW" subtitle="What happens after you submit"><AgentTimeline hasResult={false} /></Panel>
      <Panel title="SUPPORTED INPUTS" subtitle="LLM canonical fields"><div className="tag-cloud">{['budget','RAM','SSD','delivery','category','brand','battery','weight','monitor size','warranty','printer stock','reliability','defect rate','risk tolerance','weights'].map(x=><span key={x}>{x}</span>)}</div></Panel>
      <div className="info-card"><span>ⓘ</span><div><strong>Prototype environment</strong><p>The backend dataset and offer relationships are synthetic/derived. This UI is designed for demonstration and hackathon presentation.</p></div></div>
    </aside>
  </div>;
}
function Field({label,children}) { return <label className="field"><span>{label}</span>{children}</label>; }

function VendorsPage({offers,recommendation,selected,setSelected}) {
  return <div><div className="page-intro"><div><div className="tag">STAGE 2 · VENDOR INTELLIGENCE</div><h2>Compare every eligible offer</h2><p>Ranking combines price, specs fit, delivery reliability, vendor trust, warranty and price stability.</p></div><div className="legend"><span className="safe-dot"></span> Low risk <span className="warn-dot"></span> Moderate <span className="danger-dot"></span> High</div></div>
    {!offers.length ? <div className="big-empty"><div>◈</div><h3>No vendor data yet</h3><p>Run a procurement request first. The backend's <code>/api/v1/orchestrate</code> response will populate this view.</p></div> : <div className="vendor-layout"><section className="table-panel"><div className="table-wrap"><table><thead><tr><th>#</th><th>Vendor</th><th>Product</th><th>Price</th><th>Delivery</th><th>Risk</th><th>Score</th></tr></thead><tbody>{offers.map((o,i)=><tr className={selected===o?'selected':''} key={`${o.vendor_id}-${o.product_id}-${i}`} onClick={()=>setSelected(o)}><td><span className="table-rank">{i+1}</span></td><td><strong>{o.vendor_name || o.vendor_id}</strong><small>{o.vendor_id}</small></td><td><strong>{o.brand} {o.product_id}</strong><small>{o.ram}GB RAM · {o.ssd_size}</small></td><td>{money(o.offer_price)}</td><td>{o.delivery_days}d</td><td><span className={`risk ${riskClass(o.ai_risk_level)}`}>{o.ai_risk_level || '—'}</span></td><td><strong className="score-number">{score(o.vendor_score)}</strong></td></tr>)}</tbody></table></div></section><OfferDetail offer={selected || offers[0]} recommended={recommendation} /></div>}
  </div>;
}

function offerReasonText(offer, recommended) {
  if (!offer) return 'Select an offer to inspect its scoring components and AI risk signals.';
  if (recommended?.vendor_id === offer.vendor_id && recommended?.reason) return recommended.reason;

  const vendorScore = Number(offer.vendor_score || 0);
  const points = [
    ['price', Number(offer.price_score || 0) * 20],
    ['specs fit', Number(offer.specs_score || 0) * 15],
    ['delivery reliability', Number(offer.delivery_reliability_score || 0) * 25],
    ['vendor trust', Number(offer.vendor_trust_score || 0) * 20],
    ['warranty', Number(offer.warranty_score || 0) * 15],
    ['price stability', Number(offer.price_stability_score || 0) * 5],
  ].sort((a, b) => b[1] - a[1]);

  const top = points.slice(0, 3).map(([name, pts]) => `${name} (${pts.toFixed(1)} pts)`).join(', ');
  const riskLabel = offer.ai_risk_level || 'Unknown';
  const riskScore = Number(offer.ai_risk_score || 0).toFixed(2);
  const delay = Number(offer.ai_expected_delay_days || 0).toFixed(1);

  return `Score ${vendorScore.toFixed(2)}/100. Strongest components: ${top}. AI risk: ${riskLabel} (score ${riskScore}), predicted delivery deviation ~ ${delay} days. [Synthetic prototype offer - not a real quotation.]`;
}

function OfferDetail({offer,recommended}) {
  if (!offer) return null;
  const components = [['Price',offer.price_score],['Specs fit',offer.specs_score],['Delivery reliability',offer.delivery_reliability_score],['Vendor trust',offer.vendor_trust_score],['Warranty',offer.warranty_score],['Price stability',offer.price_stability_score]];
  return <aside className="offer-detail"><div className="detail-kicker">{recommended?.vendor_id === offer.vendor_id ? '★ RECOMMENDED OFFER' : 'OFFER DETAILS'}</div><h3>{offer.vendor_name || offer.vendor_id}</h3><p>{offer.brand} {offer.product_id}</p><div className="detail-score"><strong>{score(offer.vendor_score)}</strong><span>/100</span><span className={`risk ${riskClass(offer.ai_risk_level)}`}>{offer.ai_risk_level}</span></div><div className="detail-grid"><span>Price<strong>{money(offer.offer_price)}</strong></span><span>Delivery<strong>{offer.delivery_days} days</strong></span><span>Warranty<strong>{offer.warranty_years} years</strong></span><span>AI risk score<strong>{pct(offer.ai_risk_score)}</strong></span><span>Expected delay<strong>{Number(offer.ai_expected_delay_days || 0).toFixed(1)} days</strong></span><span>Global rank<strong>#{offer.rank_global || '—'}</strong></span></div><div className="component-title">Six-factor breakdown</div>{components.map(([l,v])=><div className="component" key={l}><div><span>{l}</span><b>{pct(v)}</b></div><div className="bar"><i style={{width:`${Math.max(0,Math.min(1,Number(v)||0))*100}%`}} /></div></div>)}<div className="reason">{offerReasonText(offer, recommended)}</div></aside>;
}

function WhatIfPage({offers,recommendation}) {
  const [a,setA] = useState(offers[0]?.vendor_id || '');
  const [b,setB] = useState(offers[1]?.vendor_id || '');
  useEffect(()=>{ if (offers[0]) setA(offers[0].vendor_id); if (offers[1]) setB(offers[1].vendor_id); },[offers]);
  const left = offers.find(o=>o.vendor_id===a) || offers[0];
  const right = offers.find(o=>o.vendor_id===b) || offers[1];
  if (!offers.length) return <div className="big-empty"><div>◇</div><h3>What-If needs ranked offers</h3><p>Run a procurement request, then compare two offers from the returned Stage 2 shortlist.</p></div>;
  return <div><div className="page-intro"><div><div className="tag">DECISION SIMULATOR</div><h2>Compare the trade-offs</h2><p>Use the ranked offers as scenarios. This frontend view does not change backend ranking; it visualizes the returned scores and signals.</p></div></div><div className="compare-controls"><label>Scenario A<select value={a} onChange={e=>setA(e.target.value)}>{offers.map(o=><option key={o.vendor_id}>{o.vendor_id}</option>)}</select></label><div className="vs">VS</div><label>Scenario B<select value={b} onChange={e=>setB(e.target.value)}>{offers.map(o=><option key={o.vendor_id}>{o.vendor_id}</option>)}</select></label></div><div className="scenario-grid"><ScenarioCard title="SCENARIO A" offer={left} recommended={recommendation?.vendor_id===left?.vendor_id}/><ScenarioCard title="SCENARIO B" offer={right} recommended={recommendation?.vendor_id===right?.vendor_id}/></div><div className="compare-note">ⓘ A future backend extension can expose a dedicated what-if endpoint. For now this module compares the actual ranked offers returned by your existing API.</div></div>;
}
function ScenarioCard({title,offer,recommended}) { if(!offer) return null; const rows=[['Upfront price',money(offer.offer_price)],['Vendor score',`${score(offer.vendor_score)}/100`],['Delivery',`${offer.delivery_days} days`],['AI risk',offer.ai_risk_level],['Risk score',pct(offer.ai_risk_score)],['Expected delay',`${Number(offer.ai_expected_delay_days||0).toFixed(1)} days`],['Warranty',`${offer.warranty_years} years`]]; return <section className={`scenario-card ${recommended?'recommended':''}`}><div className="scenario-head"><span>{title}</span>{recommended && <b>RECOMMENDED</b>}</div><h3>{offer.vendor_name || offer.vendor_id}</h3><p>{offer.brand} {offer.product_id}</p><div className="scenario-score"><strong>{score(offer.vendor_score)}</strong><span>score</span></div>{rows.map(([k,v])=><div className="scenario-row" key={k}><span>{k}</span><strong className={k==='AI risk'?riskClass(v):''}>{v}</strong></div>)}<div className="scenario-reason">{recommended ? 'Current backend recommendation.' : 'Alternative returned by the ranking engine.'}</div></section>; }

function ApprovalsPage({result, approvalEvents = [], onAction, busy = false, currentRequestId = null}) {
  const r=result?.recommendation, req=result?.requirements || result?.orchestration?.resolved_requirements;
  const needs = r && req?.max_price && Number(r.offer_price)>Number(req.max_price);
  const canAct = Boolean(r);
  const activeDecision = approvalEvents.find(a => {
    if (!a) return false;
    const sameReq = (a.request_id || null) === (currentRequestId || null);
    const sameVendor = (a.vendor_id || null) === (r?.vendor_id || null);
    return sameReq && sameVendor;
  });
  const locked = activeDecision && (activeDecision.action === 'approve' || activeDecision.action === 'reject');
  return <div><div className="page-intro"><div><div className="tag">HUMAN-IN-THE-LOOP</div><h2>Approval Center</h2><p>Review and record an approval decision for the latest recommendation.</p></div></div>{canAct ? <div className="approval-card"><div className="approval-icon">{needs ? '!' : '✓'}</div><div className="approval-main"><div className="approval-status">{needs ? 'EXCEPTION DETECTED' : 'STANDARD REVIEW'}</div><h3>{needs ? `${r.vendor_name || r.vendor_id} exceeds the price cap` : `${r.vendor_name || r.vendor_id} is within limits`}</h3><p>{needs ? 'The latest recommendation is above the maximum price supplied in the procurement request.' : 'No policy breach detected, but decision actions remain enabled for demo and audit capture.'}</p><div className="approval-stats"><span>Authorized limit<strong>{req?.max_price !== undefined ? money(req.max_price) : '—'}</strong></span><span>Recommended offer<strong>{money(r.offer_price)}</strong></span><span>Variance<strong>{req?.max_price !== undefined ? money(Number(r.offer_price)-Number(req.max_price)) : '—'}</strong></span><span>Risk<strong className={`risk ${riskClass(r.ai_risk_level)}`}>{r.ai_risk_level}</strong></span></div>{locked ? <div className="warning-box">Final decision already recorded: <strong>{String(activeDecision.action).toUpperCase()}</strong>. Use Request revision to reopen this item.</div> : null}<div className="approval-actions"><button className="primary-btn" onClick={()=>onAction?.('approve')} disabled={busy || locked}>{busy ? 'Saving…' : (needs ? 'Approve exception' : 'Approve recommendation')}</button><button className="secondary-btn" onClick={()=>onAction?.('reject')} disabled={busy || locked}>Reject</button><button className="ghost-btn" onClick={()=>onAction?.('revise')} disabled={busy}>Request revision</button></div></div></div> : <div className="big-empty"><div>✓</div><h3>No recommendation available</h3><p>Run a procurement request to evaluate approval actions.</p></div>}<section className="panel" style={{marginTop:'14px'}}><h3>Recent actions</h3><p className="panel-sub">Saved by backend approval endpoint</p>{approvalEvents.length ? approvalEvents.slice(0,8).map(a => <div className="history-card" key={a.id}><strong>{a.id}</strong><span>{String(a.action || '').toUpperCase()} · {a.vendor_name || a.vendor_id || 'n/a'}</span><small>{a.timestamp ? new Date(a.timestamp).toLocaleString() : 'unknown time'}</small></div>) : <EmptyPanel text="No approval actions saved yet." />}</section></div>;
}

function AuditPage({result,history}) {
  const r=result?.recommendation, o=result?.orchestration;
  const events = [];
  if (result) {
    events.push(['Recommendation returned', `Status: ${result.status}`, 'NOW']);
    if (r) events.push(['Recommendation selected', `${r.vendor_name || r.vendor_id} · ${money(r.offer_price)} · score ${score(r.vendor_score)}`, 'DECISION']);
    if (o?.llm_used) events.push(['LLM normalization', `Provider: ${o.llm_provider || 'openai'} · structured intent merged`, 'AI']);
    (o?.agents || AGENTS.map(a=>a[0])).forEach((a,i)=>events.push([a, i===0?'Flexible requirement mapping':i===1?'Range / quality validation':i===2?'Policy defaults': 'Deterministic Stage 1 + Stage 2 ranking', `0${i+1}`]));
    if (o?.warnings?.length) events.push(['Warnings', o.warnings.join(' '), 'WARN']);
  }
  return <div><div className="page-intro"><div><div className="tag">EXPLAINABILITY & TRACEABILITY</div><h2>Audit Trail</h2><p>Inspect the latest orchestration metadata and browser-session request history.</p></div></div><div className="audit-layout"><section className="panel"><div className="audit-head"><div><h3>Latest execution</h3><p>{result ? 'Backend response reconstructed into an auditable timeline.' : 'No execution yet.'}</p></div>{o && <span className="confidence-pill">CONFIDENCE {pct(o.confidence)}</span>}</div>{events.length ? <div className="audit-list">{events.map(([a,b,c],i)=><div className="audit-item" key={`${a}-${i}`}><div className="audit-line"></div><div className="audit-marker">{i===0?'●':'✓'}</div><div className="audit-copy"><strong>{a}</strong><p>{b}</p></div><span>{c}</span></div>)}</div> : <EmptyPanel text="Run a request to create an execution timeline." />}</section><section className="panel"><h3>Session history</h3><p className="panel-sub">Stored locally in this browser</p>{history.length ? history.map(h=><div className="history-card" key={h.id}><strong>{h.id}</strong><span>{h.recommendation?.vendor_name || 'No match'}</span><small>{new Date(h.time).toLocaleString()}</small></div>) : <EmptyPanel text="Nothing stored yet." />}</section></div></div>;
}

export default App;
