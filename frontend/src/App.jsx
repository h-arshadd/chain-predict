import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { themeConfig } from './theme';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <ConfigProvider theme={themeConfig}>
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<Dashboard />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;