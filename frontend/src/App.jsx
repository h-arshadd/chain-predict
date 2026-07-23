import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { themeConfig } from './theme';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Strategies from './pages/Strategies';
import StrategyDetails from './pages/StrategyDetails';

function App() {
  return (
    <ConfigProvider theme={themeConfig}>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/strategies" element={<Strategies />} />
            <Route path="/strategies/:id" element={<StrategyDetails />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;