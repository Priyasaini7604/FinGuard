const API_BASE_URL = "http://localhost:8000";

export async function fetchTransactionGraph(limit = 100) {
  const response = await fetch(`${API_BASE_URL}/transactions/graph/data?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch graph data: ${response.status}`);
  }


  return response.json();
}
export async function fetchTransactionStats() {
  const response = await fetch(`${API_BASE_URL}/transactions/stats/summary`);

  if (!response.ok) {
    throw new Error(`Failed to fetch stats: ${response.status}`);
  }

  return response.json();
}
export async function fetchFlaggedTransactions(limit = 20) {
  const response = await fetch(
    `${API_BASE_URL}/transactions/?flagged_only=true&limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch transactions: ${response.status}`);
  }

  return response.json();
}