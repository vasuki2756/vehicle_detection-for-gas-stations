import React, { useState } from 'react';
import VideoFeed from './VideoFeed';
import VehicleTable from './VehicleTable';
import AlertsPanel from './AlertsPanel';
import Footer from './Footer';
import { useWebSocket } from '../context/WebSocketContext';
import './DashboardLayout.css';

const DashboardLayout = () => {
  const [selectedVehicleId, setSelectedVehicleId] = useState(null);
  const { connectionStatus } = useWebSocket();

  const handleVehicleSelect = (trackId) => {
    setSelectedVehicleId(trackId === selectedVehicleId ? null : trackId);
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-title">
            <h1>🚛 Hazardous Vehicle Monitoring System</h1>
            <p className="header-subtitle">Real-time tracking and safety compliance</p>
          </div>
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-indicator"></span>
            {connectionStatus === 'connected' ? 'Connected' : 
             connectionStatus === 'connecting' ? 'Connecting...' : 
             'Disconnected'}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-main">
        {/* Video Feed - Left Section */}
        <section className="video-section">
          <VideoFeed 
            selectedVehicleId={selectedVehicleId}
            onVehicleHover={(vehicle) => {
              // Optional: Add hover effects
            }}
          />
        </section>

        {/* Middle Section - Vehicle Table */}
        <section className="table-section">
          <VehicleTable 
            onVehicleSelect={handleVehicleSelect}
            selectedVehicleId={selectedVehicleId}
          />
        </section>

        {/* Right Section - Alerts */}
        <section className="alerts-section">
          <AlertsPanel />
        </section>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default DashboardLayout;
