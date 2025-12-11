# Hazardous Vehicle Monitoring Dashboard

A real-time React-based dashboard for monitoring hazardous vehicles with SAMv3 mask overlays, live vehicle tracking, and alert management.

## Features

### 🎥 Live Video Feed
- Real-time video display with vehicle mask overlays from SAMv3
- Color-coded vehicle tracking:
  - **Red**: Delayed hazardous vehicles
  - **Yellow**: Unauthorized vehicles
  - **Green**: Worker/authorized vehicles
  - **Orange**: Normal hazardous vehicles
- Interactive hover tooltips showing vehicle details
- Zone boundary overlays
- Track ID labels on each vehicle

### 📊 Vehicle Table
- Comprehensive vehicle log with sortable columns
- Real-time updates via WebSocket
- Filter by:
  - Vehicle type (Hazardous/Worker/Other)
  - Status (In Transit/Delayed/Stored/Unauthorized)
- Click to highlight vehicle on video feed
- Color-coded rows for quick status identification

### 🚨 Alerts Panel
- Live alert cards with priority indicators
- Alert types:
  - Delays
  - Unauthorized vehicles
  - Zone deviations
- One-click alert acknowledgment
- Filtered alert views

### 📈 Footer Metrics
- Active hazardous vehicle count
- Unauthorized vehicle count
- Delayed vehicle count
- Total active alerts

## Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Configure environment variables:
Create a `.env` file in the frontend directory:
```
REACT_APP_SOCKET_URL=http://localhost:8000
```

3. Start the development server:
```bash
npm start
```

The dashboard will open at `http://localhost:3000`

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── VideoFeed.js          # Video with SAM mask overlays
│   │   ├── VideoFeed.css
│   │   ├── VehicleTable.js       # Vehicle log table
│   │   ├── VehicleTable.css
│   │   ├── AlertsPanel.js        # Live alerts
│   │   ├── AlertsPanel.css
│   │   ├── DashboardLayout.js    # Main layout
│   │   ├── DashboardLayout.css
│   │   ├── Footer.js             # Stats footer
│   │   └── Footer.css
│   ├── context/
│   │   └── WebSocketContext.js   # WebSocket management
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

## WebSocket Events

### Client Listens For:
- `vehicle_update`: Bulk vehicle data updates
- `vehicle_position`: Individual vehicle position updates
- `alert`: New alert notifications
- `zones_update`: Zone configuration updates
- `alert_acknowledged`: Confirmation of alert acknowledgment

### Client Emits:
- `request_initial_data`: Request current state
- `acknowledge_alert`: Acknowledge an alert

## Backend Integration

The frontend expects WebSocket events with the following data structures:

### Vehicle Data
```javascript
{
  track_id: "123",
  vehicle_type: "Hazardous",
  plate_number: "ABC-1234",
  current_zone: "Loading Zone",
  time_in_zone: 120,
  status: "in_transit",
  mask: [[x1, y1], [x2, y2], ...],
  entry_time: "2025-12-11T10:30:00Z"
}
```

### Alert Data
```javascript
{
  id: "alert_123",
  timestamp: "2025-12-11T10:35:00Z",
  track_id: "123",
  plate_number: "ABC-1234",
  alert_type: "Delay",
  message: "Vehicle exceeded maximum time in loading zone"
}
```

### Zone Data
```javascript
{
  name: "Loading Zone",
  polygon: [[x1, y1], [x2, y2], ...]
}
```

## Customization

### Colors
Edit CSS files to customize the color scheme:
- Primary colors defined in component CSS files
- Dark theme: `#0f172a`, `#1e293b`
- Accent colors: Red (`#ef4444`), Yellow (`#eab308`), Green (`#22c55e`)

### Video Source
Update the video source in `VideoFeed.js`:
```javascript
<source src="/your-video.mp4" type="video/mp4" />
```

Or connect to a live stream:
```javascript
<video src="http://your-stream-url" />
```

## Browser Support
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Performance Notes
- Canvas rendering optimized for 60 FPS
- WebSocket reconnection with exponential backoff
- Alert list limited to 50 most recent
- Efficient polygon hit detection for hover

## Development

Build for production:
```bash
npm run build
```

Run tests:
```bash
npm test
```

## License
MIT
