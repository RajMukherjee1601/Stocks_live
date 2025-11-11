import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

// ========= Backend =========
const API_BASE = "http://localhost:5000/api/stocks"; // change if needed

// ========= Stocks =========
const NAME_TO_TICKER = {
  "reliance": "RELIANCE.NS",
  "tata steel": "TATASTEEL.NS",
  "hdfc bank": "HDFCBANK.NS",
  "tcs": "TCS.NS",
  "infosys": "INFY.NS",
  "icici bank": "ICICIBANK.NS",
  "state bank of india": "SBIN.NS",
  "axis bank": "AXISBANK.NS",
  "kotak mahindra bank": "KOTAKBANK.NS",
  "hindustan unilever": "HINDUNILVR.NS",
  "itc": "ITC.NS",
  "larsen and toubro": "LT.NS",
  "bharti airtel": "BHARTIARTL.NS",
  "asian paints": "ASIANPAINT.NS",
  "hcl technologies": "HCLTECH.NS",
  "maruti suzuki": "MARUTI.NS",
  "bajaj finance": "BAJFINANCE.NS",
  "bajaj finserv": "BAJAJFINSV.NS",
  "power grid": "POWERGRID.NS",
  "ntpc": "NTPC.NS",
  "ongc": "ONGC.NS",
  "adani enterprises": "ADANIENT.NS",
  "adani ports": "ADANIPORTS.NS",
  "sbi life insurance": "SBILIFE.NS",
  "titan": "TITAN.NS",
  "ultratech cement": "ULTRACEMCO.NS",
  "sun pharma": "SUNPHARMA.NS",
  "wipro": "WIPRO.NS",
  "tech mahindra": "TECHM.NS",
  "dr reddy": "DRREDDY.NS",
  "cipla": "CIPLA.NS",
  "grasim": "GRASIM.NS",
  "eicher motors": "EICHERMOT.NS",
  "nestle india": "NESTLEIND.NS",
  "jsw steel": "JSWSTEEL.NS",
  "indusind bank": "INDUSINDBK.NS",
  "apollo hospitals": "APOLLOHOSP.NS",
  "tata motors": "TATAMOTORS.NS",
  "coal india": "COALINDIA.NS",
  "heromotocorp": "HEROMOTOCO.NS",
  "britannia": "BRITANNIA.NS",
  "divi labs": "DIVISLAB.NS",
  "hindalco": "HINDALCO.NS",
  "mahanagar gas": "MGL.NS",
  "upl": "UPL.NS",
  "sbi cards": "SBICARD.NS",
};

// ========= Timeframes & yfinance-compatible params =========
const TF = {
  "1D": { period: "1d", interval: "5m" },
  "1M": { period: "1mo", interval: "60m" },
  "1Y": { period: "1y", interval: "1d" },
  "3Y": { period: "3y", interval: "1d" },
};

const fmtTime = (iso) => new Date(iso).toLocaleString("en-IN", { hour12: true });

export default function RealTimeStock() {
  const [symbol, setSymbol] = useState("RELIANCE.NS");
  const [tf, setTf] = useState("1D");
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const title = useMemo(() => {
    const name = Object.entries(NAME_TO_TICKER).find(([, t]) => t === symbol)?.[0] || symbol;
    return `${name} (${symbol})`;
  }, [symbol]);

  async function fetchData() {
    setLoading(true);
    setError("");
    try {
      const { period, interval } = TF[tf];
      const res = await axios.get(`${API_BASE}/intraday`, { params: { symbol, period, interval } });
      const rows = (res.data?.data || []).map((d) => ({ time: d.timestamp, price: d.price }));
      setData(rows);
      setLastUpdated(new Date().toISOString());
    } catch (e) {
      setError(e?.response?.data?.error || e.message);
      setData([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, tf]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#0f172a", color: "#e2e8f0" }}>
      {/* Header / Controls */}
      <div style={{ padding: "12px 16px", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, marginRight: 12 }}>NSE Real‑Time Viewer</h1>

        {/* Stock dropdown */}
        <label style={{ fontSize: 12 }}>Stock:&nbsp;
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 10 }}
          >
            {Object.entries(NAME_TO_TICKER).map(([name, ticker]) => (
              <option key={ticker} value={ticker}>{name}</option>
            ))}
          </select>
        </label>

        {/* Timeframe buttons */}
        <div style={{ display: "flex", gap: 8 }}>
          {Object.keys(TF).map((k) => (
            <button
              key={k}
              onClick={() => setTf(k)}
              disabled={loading}
              style={{
                padding: "6px 12px",
                borderRadius: 10,
                border: "1px solid #475569",
                background: tf === k ? "#22c55e" : "transparent",
                color: tf === k ? "#0b1b13" : "#e2e8f0",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {k}
            </button>
          ))}
        </div>

        {/* Refresh */}
        <button onClick={fetchData} disabled={loading} style={{
          padding: "6px 12px",
          borderRadius: 10,
          border: "1px solid #475569",
          background: "transparent",
          color: "#e2e8f0",
          cursor: loading ? "not-allowed" : "pointer",
        }}>
          {loading ? "Loading…" : "Refresh"}
        </button>

        {/* Last updated */}
        {lastUpdated && (
          <span style={{ fontSize: 12, opacity: 0.8 }}>Updated: {fmtTime(lastUpdated)}</span>
        )}

        {/* Error */}
        {error && (
          <span style={{ fontSize: 12, color: "#fecaca" }}>Error: {error}</span>
        )}
      </div>

      {/* Full‑screen chart area */}
      <div style={{ width: "100%", height: "calc(100vh - 64px)", background: "#111827" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#94a3b8" }} tickLine={false} axisLine={{ stroke: "#1f2937" }} />
            <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} tickLine={false} axisLine={{ stroke: "#1f2937" }} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151", color: "#e5e7eb" }} />
            <Legend wrapperStyle={{ color: "#e5e7eb" }} />
            <Line type="monotone" dataKey="price" name={title} stroke="#60a5fa" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
