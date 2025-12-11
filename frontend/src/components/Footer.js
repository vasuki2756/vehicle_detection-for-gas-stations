import React from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import './Footer.css';

const Footer = () => {
  const { vehicleData, alerts } = useWebSocket();

  const stats = {
    activeHazardous: vehicleData ? vehicleData.filter(v => v.vehicle_type === 'Hazardous').length : 0,
    unauthorized: vehicleData ? vehicleData.filter(v => v.status === 'unauthorized').length : 0,
    delays: vehicleData ? vehicleData.filter(v => v.status === 'delayed').length : 0,
    totalAlerts: alerts ? alerts.length : 0
  };

  return (
    <footer className="dashboard-footer">
      <div className="footer-content">
        <div className="footer-stats">
          <div className="stat-item">
            <span className="stat-label">Active Hazardous Vehicles</span>
            <span className="stat-value hazardous">{stats.activeHazardous}</span>
          </div>
          <div className="stat-separator"></div>
          <div className="stat-item">
            <span className="stat-label">Unauthorized Vehicles</span>
            <span className="stat-value unauthorized">{stats.unauthorized}</span>
          </div>
          <div className="stat-separator"></div>
          <div className="stat-item">
            <span className="stat-label">Delayed Vehicles</span>
            <span className="stat-value delayed">{stats.delays}</span>
          </div>
          <div className="stat-separator"></div>
          <div className="stat-item">
            <span className="stat-label">Total Alerts</span>
            <span className="stat-value alerts">{stats.totalAlerts}</span>
          </div>
        </div>
        
        <div className="footer-info">
          <p>© 2025 Hazardous Vehicle Monitoring System | Real-time Safety Compliance</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
