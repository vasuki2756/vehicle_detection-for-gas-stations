import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [vehicleData, setVehicleData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [zones, setZones] = useState([]);

  // Connect to WebSocket server
  useEffect(() => {
    const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'ws://localhost:8000';
    
    setConnectionStatus('connecting');
    
    // Use native WebSocket instead of Socket.IO for better sync
    const newSocket = new WebSocket(`${SOCKET_URL}/ws`);

    newSocket.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    newSocket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
    };

    newSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('disconnected');
    };

    newSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle different message types
        switch (data.type) {
          case 'vehicle_update':
            setVehicleData(data.vehicles || []);
            break;
            
          case 'alert':
            setAlerts(prev => {
              const exists = prev.some(a => a.id === data.id);
              if (exists) return prev;
              return [data, ...prev].slice(0, 50);
            });
            break;
            
          case 'zones_update':
            setZones(data.zones || []);
            break;
            
          case 'vehicle_position':
            setVehicleData(prev => {
              const index = prev.findIndex(v => v.track_id === data.track_id);
              if (index !== -1) {
                const updated = [...prev];
                updated[index] = { ...updated[index], ...data };
                return updated;
              }
              return [...prev, data];
            });
            break;
            
          case 'alert_acknowledged':
            setAlerts(prev => prev.filter(a => a.id !== data.alert_id));
            break;
            
          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    setSocket(newSocket);

    return () => {
      if (newSocket.readyState === WebSocket.OPEN) {
        newSocket.close();
      }
    };
  }, []);

  // Function to acknowledge an alert
  const acknowledgeAlert = useCallback((alertId) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      console.log('Acknowledging alert:', alertId);
      socket.send(JSON.stringify({
        type: 'acknowledge_alert',
        alert_id: alertId
      }));
      
      // Optimistically remove from UI
      setAlerts(prev => prev.filter(a => a.id !== alertId));
    }
  }, [socket]);

  // Function to manually request data refresh
  const refreshData = useCallback(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'request_initial_data' }));
    }
  }, [socket]);

  const value = {
    socket,
    connectionStatus,
    vehicleData,
    alerts,
    zones,
    acknowledgeAlert,
    refreshData
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
