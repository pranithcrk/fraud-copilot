import { useState, useEffect } from "react";

const API = "http://localhost:8000";

const riskColors = { HIGH: "#ef4444", MEDIUM: "#f59e0b", LOW: "#22c55e" };

export default function App() {
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [custom, setCustom] = useState({ amount: "", time: "" });

  useEffect(() => {
    fetch(`${API}/samples`).then((r) => r.json()).then(setSamples).catch(() => {});
  }, []);

  async function analyze(txn) {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(txn),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: e.message });
    }
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 24, fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700 }}>Fraud Intelligence Copilot</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>
        ML fraud scoring + LLM explanations powered by local Ollama
      </p>

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>Sample Transactions</h2>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
        {samples.map((s, i) => (
          <button
            key={i}
            onClick={() => analyze(s)}
            style={{
              padding: "10px 16px",
              border: "1px solid #ddd",
              borderRadius: 8,
              background: "#fff",
              cursor: "pointer",
              fontSize: 14,
            }}
          >
            {s.label} — ${s.amount}
          </button>
        ))}
      </div>

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>Custom Transaction</h2>
      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          placeholder="Amount"
          type="number"
          value={custom.amount}
          onChange={(e) => setCustom({ ...custom, amount: e.target.value })}
          style={{ padding: 8, border: "1px solid #ddd", borderRadius: 6, width: 120 }}
        />
        <input
          placeholder="Time (sec)"
          type="number"
          value={custom.time}
          onChange={(e) => setCustom({ ...custom, time: e.target.value })}
          style={{ padding: 8, border: "1px solid #ddd", borderRadius: 6, width: 120 }}
        />
        <button
          onClick={() => analyze({ amount: +custom.amount, time: +custom.time })}
          style={{
            padding: "8px 16px",
            background: "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          Analyze
        </button>
      </div>

      {loading && <p>Analyzing...</p>}

      {result && !result.error && (
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 12,
            padding: 20,
            background: "#fafafa",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 14, color: "#666" }}>Fraud Score</div>
              <div style={{ fontSize: 32, fontWeight: 700 }}>{(result.fraud_score * 100).toFixed(1)}%</div>
            </div>
            <div
              style={{
                padding: "6px 16px",
                borderRadius: 20,
                background: riskColors[result.risk_level] + "22",
                color: riskColors[result.risk_level],
                fontWeight: 600,
                alignSelf: "center",
              }}
            >
              {result.risk_level} RISK
            </div>
          </div>

          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Top Features</h3>
          {result.top_features.map((f, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, padding: "4px 0" }}>
              <span>{f.name}</span>
              <span>
                value: {f.value} | importance: {f.importance}
              </span>
            </div>
          ))}

          <h3 style={{ fontSize: 14, fontWeight: 600, marginTop: 16, marginBottom: 8 }}>AI Explanation</h3>
          <p style={{ fontSize: 14, lineHeight: 1.6, color: "#333" }}>{result.explanation}</p>
        </div>
      )}

      {result?.error && (
        <p style={{ color: "#ef4444" }}>Error: {result.error}</p>
      )}
    </div>
  );
}
