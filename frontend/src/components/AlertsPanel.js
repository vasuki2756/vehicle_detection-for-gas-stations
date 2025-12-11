import React, { useState } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import './AlertsPanel.css';

const AlertsPanel = () => {
  const { alerts, acknowledgeAlert } = useWebSocket();
  const [filter, setFilter] = useState('all');

  const filteredAlerts = alerts ? alerts.filter(alert => {
    if (filter === 'all') return true;
    return alert.alert_type === filter;
  }) : [];

  const getAlertClassName = (alertType) => {
    switch (alertType) {
      case 'Delay':
        return 'alert-danger';
      case 'Unauthorized':
        return 'alert-warning';
      case 'Zone deviation':
        return 'alert-info';
      default:
        return 'alert-default';
    }
  };

  const getAlertIcon = (alertType) => {
    switch (alertType) {
      case 'Delay':
        return '⚠️';
      case 'Unauthorized':
        return '🚫';
      case 'Zone deviation':
        return '📍';
      default:
        return '🔔';
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const handleAcknowledge = (alertId) => {
    if (acknowledgeAlert) {
      acknowledgeAlert(alertId);
    }
  };

  return (
    <div className="alerts-panel">
      <div className="alerts-header">
        <h2>Alerts</h2>
        <div className="alert-count">
          {filteredAlerts.length} active
        </div>
      </div>

      {/* Filter */}
      <div className="alerts-filter">
        <button
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All
        </button>
        <button
          className={filter === 'Delay' ? 'active' : ''}
          onClick={() => setFilter('Delay')}
        >
          Delays
        </button>
        <button
          className={filter === 'Unauthorized' ? 'active' : ''}
          onClick={() => setFilter('Unauthorized')}
        >
          Unauthorized
        </button>
        <button
          className={filter === 'Zone deviation' ? 'active' : ''}
          onClick={() => setFilter('Zone deviation')}
        >
          Deviations
        </button>
      </div>

      {/* Alerts List */}
      <div className="alerts-list">
        {filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            <div className="no-alerts-icon">✓</div>
            <p>No active alerts</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`alert-card ${getAlertClassName(alert.alert_type)}`}
            >
              <div className="alert-icon">
                {getAlertIcon(alert.alert_type)}
              </div>
              
              <div className="alert-content">
                <div className="alert-header-row">
                  <span className="alert-type">{alert.alert_type}</span>
                  <span className="alert-time">{formatTime(alert.timestamp)}</span>
                </div>
                
                <div className="alert-details">
                  {alert.track_id && (
                    <div className="alert-detail">
                      <strong>Track ID:</strong> {alert.track_id}
                    </div>
                  )}
                  {alert.plate_number && (
                    <div className="alert-detail">
                      <strong>Plate:</strong> {alert.plate_number}
                    </div>
                  )}
                </div>
                
                <div className="alert-message">
                  {alert.message}
                </div>
              </div>

              <button
                className="acknowledge-btn"
                onClick={() => handleAcknowledge(alert.id)}
                title="Acknowledge alert"
              >
                ✓
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertsPanel;
