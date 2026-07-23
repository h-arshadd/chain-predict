import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { themeConfig } from './theme';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Strategies from './pages/Strategies';
import StrategyDetails from './pages/StrategyDetails';
import Wallets from './pages/Wallets';
import Deployment from './pages/Deployment';
import ExecutionDetails from './pages/ExecutionDetails';
import Backtests from './pages/Backtests';
import BacktestDetails from './pages/BacktestDetails';
import Models from './pages/Models';
import ModelDetails from './pages/ModelDetails';
import Sentiment from './pages/Sentiment';

function App() {
  return (
    <ConfigProvider theme={themeConfig}>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/strategies" element={<Strategies />} />
            <Route path="/strategies/:id" element={<StrategyDetails />} />
            <Route path="/wallets" element={<Wallets />} />
            <Route path="/deployment" element={<Deployment />} />
            <Route path="/deployment/:id" element={<ExecutionDetails />} />
            <Route path="/backtests" element={<Backtests />} />
            <Route path="/backtests/:id" element={<BacktestDetails />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelDetails />} />
            <Route path="/sentiment" element={<Sentiment />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;