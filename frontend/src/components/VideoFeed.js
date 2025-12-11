import React, { useRef, useEffect, useState } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import './VideoFeed.css';

const VideoFeed = ({ selectedVehicleId, onVehicleHover }) => {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const animationFrameRef = useRef(null);
  const { vehicleData, zones } = useWebSocket();
  const [hoveredVehicle, setHoveredVehicle] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update timestamp every second for CCTV effect
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Draw vehicle masks and overlays - runs on every vehicle data update
  useEffect(() => {
    const drawFrame = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      if (!canvas) return;

      const ctx = canvas.getContext('2d');

      // Match canvas size to video
      if (video && video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw zones (optional mini-map style)
      if (zones && zones.length > 0) {
        zones.forEach(zone => {
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.5)';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          if (zone.polygon && zone.polygon.length > 0) {
            ctx.moveTo(zone.polygon[0][0], zone.polygon[0][1]);
            zone.polygon.forEach(point => {
              ctx.lineTo(point[0], point[1]);
            });
            ctx.closePath();
            ctx.stroke();

            // Zone label
            ctx.fillStyle = 'rgba(100, 116, 139, 0.8)';
            ctx.font = '14px sans-serif';
            ctx.fillText(zone.name, zone.polygon[0][0] + 5, zone.polygon[0][1] - 5);
          }
          ctx.setLineDash([]);
        });
      }

      // Draw vehicle masks
      if (vehicleData && vehicleData.length > 0) {
        vehicleData.forEach(vehicle => {
          // Determine color based on vehicle type and status
          let color;
          if (vehicle.status === 'delayed' && vehicle.vehicle_type === 'Hazardous') {
            color = '#ef4444'; // Red
          } else if (vehicle.status === 'unauthorized') {
            color = '#eab308'; // Yellow
          } else if (vehicle.vehicle_type === 'Worker') {
            color = '#22c55e'; // Green
          } else if (vehicle.vehicle_type === 'Hazardous') {
            color = '#f97316'; // Orange
          } else {
            color = '#3b82f6'; // Blue
          }

          // Highlight selected vehicle
          if (vehicle.track_id === selectedVehicleId) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 20;
          }

          // Draw mask polygon
          if (vehicle.mask && vehicle.mask.length > 0) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(vehicle.mask[0][0], vehicle.mask[0][1]);
            vehicle.mask.forEach(point => {
              ctx.lineTo(point[0], point[1]);
            });
            ctx.closePath();
            ctx.stroke();

            // Fill with semi-transparent color
            ctx.fillStyle = color + '20';
            ctx.fill();

            ctx.shadowBlur = 0;

            // Draw track ID label
            const centerX = vehicle.mask.reduce((sum, p) => sum + p[0], 0) / vehicle.mask.length;
            const centerY = vehicle.mask.reduce((sum, p) => sum + p[1], 0) / vehicle.mask.length;

            ctx.fillStyle = color;
            ctx.fillRect(centerX - 30, centerY - 15, 60, 25);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`ID: ${vehicle.track_id}`, centerX, centerY + 3);
            ctx.textAlign = 'left';
          }
        });
      }
    };
    
    drawFrame();
    
    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [vehicleData, zones, selectedVehicleId]);

  // Handle mouse hover
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setMousePos({ x: e.clientX, y: e.clientY });

    // Check if mouse is over any vehicle
    let found = null;
    if (vehicleData) {
      for (const vehicle of vehicleData) {
        if (vehicle.mask && isPointInPolygon([x, y], vehicle.mask)) {
          found = vehicle;
          break;
        }
      }
    }

    setHoveredVehicle(found);
    if (onVehicleHover) {
      onVehicleHover(found);
    }
  };

  const isPointInPolygon = (point, polygon) => {
    const [x, y] = point;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i];
      const [xj, yj] = polygon[j];
      const intersect = ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  return (
    <div className="video-feed">
      <div className="video-header">
        <h2>Live Video Feed</h2>
        <div className="status-indicator">
          <span className="status-dot"></span>
          Live
        </div>
      </div>
      <div className="video-container">
        {/* CCTV-style timestamp overlay */}
        <div className="cctv-overlay">
          CAM 01 | {currentTime.toLocaleDateString('en-US', { 
            month: '2-digit', 
            day: '2-digit', 
            year: 'numeric' 
          })} {currentTime.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
          })}
        </div>
        <video
          ref={videoRef}
          className="video-element"
          autoPlay
          muted
          loop
          playsInline
          onLoadedMetadata={() => {
            // Initialize canvas size when video loads
            const canvas = canvasRef.current;
            const video = videoRef.current;
            if (canvas && video) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
            }
          }}
        >
          <source src="/sample-video.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <canvas
          ref={canvasRef}
          className="overlay-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredVehicle(null)}
        />
        
        {/* Hover tooltip */}
        {hoveredVehicle && (
          <div
            className="hover-tooltip"
            style={{
              left: mousePos.x + 10,
              top: mousePos.y + 10
            }}
          >
            <div className="tooltip-row">
              <strong>Track ID:</strong> {hoveredVehicle.track_id}
            </div>
            <div className="tooltip-row">
              <strong>Type:</strong> {hoveredVehicle.vehicle_type}
            </div>
            {hoveredVehicle.plate_number && (
              <div className="tooltip-row">
                <strong>Plate:</strong> {hoveredVehicle.plate_number}
              </div>
            )}
            <div className="tooltip-row">
              <strong>Zone:</strong> {hoveredVehicle.current_zone || 'Unknown'}
            </div>
            <div className="tooltip-row">
              <strong>Entry:</strong> {hoveredVehicle.entry_time ? new Date(hoveredVehicle.entry_time).toLocaleTimeString() : 'N/A'}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="legend">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#ef4444' }}></span>
            Hazardous (Delayed)
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#eab308' }}></span>
            Unauthorized
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#22c55e' }}></span>
            Worker/Authorized
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#f97316' }}></span>
            Hazardous (Normal)
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoFeed;
