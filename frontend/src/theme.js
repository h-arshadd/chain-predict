export const themeConfig = {
  token: {
    colorPrimary: '#6C5CE7',
    colorBgBase: '#F7F8FA',
    colorTextBase: '#1D2129',
    colorSuccess: '#12B76A',
    colorError: '#F04438',
    borderRadius: 10,
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
  },
  components: {
    Layout: {
      siderBg: '#FFFFFF',
      headerBg: '#FFFFFF',
      bodyBg: '#F7F8FA',
    },
    Menu: {
      itemSelectedBg: 'rgba(108, 92, 231, 0.12)',
      itemSelectedColor: '#6C5CE7',
      itemHoverBg: 'rgba(108, 92, 231, 0.06)',
      itemHeight: 44,
    },
    Card: {
      borderRadiusLG: 18,
    },
    Table: {
      headerBg: '#FAFAFB',
      headerColor: '#6B7280',
      borderColor: '#F0F0F2',
    },
  },
};