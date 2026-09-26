import { useEffect, useState } from "react";
import { fetchFlaggedTransactions } from "../services/api";

const thStyle = {
  textAlign: "left",
  padding: "10px 14px",
  color: "#8899a6",
  fontSize: "12px",
  textTransform: "uppercase",
  borderBottom: "1px solid #1e2733",
};

const tdStyle = {
  padding: "10px 14px",
  borderBottom: "1px solid #161d27",
  color: "#e6edf3",
  fontSize: "14px",
};

export default function FlaggedTransactionsTable() {
  const [transactions, setTransactions] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFlaggedTransactions(20)
      .then((data) => setTransactions(data.transactions))
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p style={{ color: "#ff6b6b" }}>Failed to load transactions: {error}</p>;

  return (
    <div style={{ marginTop: "24px" }}>
      <h2 style={{ color: "#e6edf3", fontSize: "18px", marginBottom: "12px" }}>
        Flagged Transactions
      </h2>
      <div style={{ background: "#131a24", borderRadius: "10px", overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={thStyle}>Transaction ID</th>
              <th style={thStyle}>Sender</th>
              <th style={thStyle}>Receiver</th>
              <th style={thStyle}>Amount</th>
              <th style={thStyle}>Type</th>
              <th style={thStyle}>Risk Score</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((txn) => (
              <tr key={txn.transaction_id}>
                <td style={tdStyle}>{txn.transaction_id}</td>
                <td style={tdStyle}>{txn.sender_account}</td>
                <td style={tdStyle}>{txn.receiver_account}</td>
                <td style={tdStyle}>₹{txn.amount.toLocaleString()}</td>
                <td style={tdStyle}>{txn.transaction_type}</td>
                <td style={{ ...tdStyle, color: "#ff5c5c", fontWeight: 600 }}>
                  {txn.risk_score.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}