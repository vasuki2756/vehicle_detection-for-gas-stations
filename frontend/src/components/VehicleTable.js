import React, { useState, useMemo } from 'react';
import { useWebSocket } from '../context/WebSocketContext';
import './VehicleTable.css';

const VehicleTable = ({ onVehicleSelect, selectedVehicleId }) => {
  const { vehicleData } = useWebSocket();
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortColumn, setSortColumn] = useState('track_id');
  const [sortDirection, setSortDirection] = useState('asc');

  // Filter and sort vehicles
  const filteredVehicles = useMemo(() => {
    if (!vehicleData) return [];

    let filtered = [...vehicleData];

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(v => v.vehicle_type === filterType);
    }

    // Filter by status
    if (filterStatus !== 'all') {
      filtered = filtered.filter(v => v.status === filterStatus);
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal = a[sortColumn];
      let bVal = b[sortColumn];

      // Handle numeric columns
      if (sortColumn === 'track_id') {
        aVal = parseInt(aVal) || 0;
        bVal = parseInt(bVal) || 0;
      }

      // Handle time columns
      if (sortColumn === 'time_in_zone') {
        aVal = parseFloat(aVal) || 0;
        bVal = parseFloat(bVal) || 0;
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [vehicleData, filterType, filterStatus, sortColumn, sortDirection]);

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const getRowClassName = (vehicle) => {
    let className = 'table-row';
    
    if (vehicle.track_id === selectedVehicleId) {
      className += ' selected';
    }

    if (vehicle.status === 'delayed' && vehicle.vehicle_type === 'Hazardous') {
      className += ' row-danger';
    } else if (vehicle.status === 'unauthorized') {
      className += ' row-warning';
    }

    return className;
  };

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const getStatusBadge = (status) => {
    const statusMap = {
      'in_transit': { label: 'In Transit', className: 'status-badge-info' },
      'delayed': { label: 'Delayed', className: 'status-badge-danger' },
      'stored': { label: 'Stored', className: 'status-badge-success' },
      'unauthorized': { label: 'Unauthorized', className: 'status-badge-warning' }
    };

    const statusInfo = statusMap[status] || { label: status, className: 'status-badge-default' };
    return <span className={`status-badge ${statusInfo.className}`}>{statusInfo.label}</span>;
  };

  return (
    <div className="vehicle-table">
      <div className="table-header">
        <h2>Vehicle Log</h2>
        <div className="vehicle-count">
          {filteredVehicles.length} vehicle{filteredVehicles.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Filters */}
      <div className="filters">
        <div className="filter-group">
          <label>Type:</label>
          <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
            <option value="all">All Types</option>
            <option value="Hazardous">Hazardous</option>
            <option value="Worker">Worker</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Status:</label>
          <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
            <option value="all">All Status</option>
            <option value="in_transit">In Transit</option>
            <option value="delayed">Delayed</option>
            <option value="stored">Stored</option>
            <option value="unauthorized">Unauthorized</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th onClick={() => handleSort('track_id')} className="sortable">
                Track ID {sortColumn === 'track_id' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th onClick={() => handleSort('vehicle_type')} className="sortable">
                Type {sortColumn === 'vehicle_type' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th>Plate</th>
              <th onClick={() => handleSort('current_zone')} className="sortable">
                Zone {sortColumn === 'current_zone' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th onClick={() => handleSort('time_in_zone')} className="sortable">
                Time in Zone {sortColumn === 'time_in_zone' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th onClick={() => handleSort('status')} className="sortable">
                Status {sortColumn === 'status' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredVehicles.length === 0 ? (
              <tr>
                <td colSpan="6" className="no-data">No vehicles found</td>
              </tr>
            ) : (
              filteredVehicles.map((vehicle) => (
                <tr
                  key={vehicle.track_id}
                  className={getRowClassName(vehicle)}
                  onClick={() => onVehicleSelect(vehicle.track_id)}
                >
                  <td className="track-id">{vehicle.track_id}</td>
                  <td>
                    <span className={`type-badge type-${vehicle.vehicle_type.toLowerCase()}`}>
                      {vehicle.vehicle_type}
                    </span>
                  </td>
                  <td className="plate-number">{vehicle.plate_number || 'N/A'}</td>
                  <td>{vehicle.current_zone || 'Unknown'}</td>
                  <td>{formatTime(vehicle.time_in_zone)}</td>
                  <td>{getStatusBadge(vehicle.status)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default VehicleTable;
