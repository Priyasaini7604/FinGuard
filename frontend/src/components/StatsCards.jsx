import { useEffect, useState } from "react";
import { fetchTransactionStats } from "../services/api";

export default function StatsCards() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTransactionStats()
      .then(setStats)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p style={{ color: "#ff6b6b" }}>Failed to load stats: {error}</p>;
  if (!stats) return <p style={{ color: "#aaa" }}>Loading stats…</p>;

  const cards = [
    { label: "Total Transactions", value: stats.total_transactions.toLocaleString() },
    { label: "Flagged Transactions", value: stats.flagged_transactions.toLocaleString(), highlight: true },
    { label: "Total Risk Amount", value: `₹${stats.total_risk_amount.toLocaleString()}`, highlight: true },
    { label: "Unique Accounts", value: stats.unique_accounts.toLocaleString() },
  ];

  return (
    <div style={{ display: "flex", gap: "16px", marginBottom: "24px", flexWrap: "wrap" }}>
      {cards.map((card) => (
        <div
          key={card.label}
          style={{
            background: "#131a24",
            border: `1px solid ${card.highlight ? "#ff3c3c55" : "#1e2733"}`,
            borderRadius: "10px",
            padding: "16px 20px",
            minWidth: "160px",
            flex: 1,
          }}
        >
          <p style={{ margin: 0, color: "#8899a6", fontSize: "13px" }}>{card.label}</p>
          <p
            style={{
              margin: "6px 0 0",
              color: card.highlight ? "#ff5c5c" : "#e6edf3",
              fontSize: "24px",
              fontWeight: 600,
            }}
          >
            {card.value}
          </p>
        </div>
      ))}
    </div>
  );
}