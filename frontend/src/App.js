import React from 'react';
import DashboardLayout from './components/DashboardLayout';
import { WebSocketProvider } from './context/WebSocketContext';
import './App.css';

function App() {
  return (
    <WebSocketProvider>
      <DashboardLayout />
    </WebSocketProvider>
  );
}

export default App;
