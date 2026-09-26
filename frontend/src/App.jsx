import TransactionGraph from "./components/TransactionGraph";
import StatsCards from "./components/StatsCards";
import FlaggedTransactionsTable from "./components/FlaggedTransactionsTable";
import "./App.css";

function App() {
  return (
    <div className="app">
      <div className="dashboard-container">
        <h1>FinGuard — AML Transaction Graph</h1>
        <StatsCards />
        <TransactionGraph />
        <FlaggedTransactionsTable />
      </div>
    </div>
  );
}

export default App;