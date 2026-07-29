import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { themeConfig } from './theme';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Wallets from './pages/Wallets';
import WalletDetails from './pages/WalletDetails';
import Execution from './pages/Execution';
import ExecutionDetails from './pages/ExecutionDetails';
import Backtests from './pages/Backtests';
import BacktestDetails from './pages/BacktestDetails';
import StrategyBuilder from './pages/StrategyBuilder';
import Models from './pages/Models';
import ModelDetails from './pages/ModelDetails';
import Sentiment from './pages/Sentiment';
import SimulationDetails from './pages/SimulationDetails';

function App() {
  return (
    <ConfigProvider theme={themeConfig}>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/wallets" element={<Wallets />} />
            <Route path="/wallets/:accountName" element={<WalletDetails />} />
            <Route path="/execution" element={<Execution />} />
            <Route path="/execution/:exchange/:symbol" element={<ExecutionDetails />} />
            <Route path="/backtests" element={<Backtests />} />
            <Route path="/backtests/:id" element={<BacktestDetails />} />
            <Route path="/strategy-builder" element={<StrategyBuilder />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelDetails />} />
            <Route path="/sentiment" element={<Sentiment />} />
            <Route path="/simulation/:exchange/:symbol/:strategyName" element={<SimulationDetails />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;