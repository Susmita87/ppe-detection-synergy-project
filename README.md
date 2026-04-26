# PPE Detection Synergy Project

A real-time Personal Protective Equipment (PPE) detection system using AI (FAST API & React Lite). This application processes images and videos to identify safety gear violations (e.g., missing helmets, vests, etc.).

## Key Features

- **Real-time Detection**: Processes images and video streams.
- **Visual Feedback**: Bounding boxes and status overlays on detected people.
- **Safety Status**: Instant "SAFE" or "VIOLATION DETECTED" indicators.
- **Web Interface**: Lightweight React-based UI for uploading and viewing results.
- **Mobile App**: Cordova-based Android app for edge device usage.
- **Docker Support**: Containerized deployment with docker-compose.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Node.js and npm (for mobile app development)
- Java JDK 11+ and Android SDK (for APK builds)

## Setup & Installation

The project includes an automated script (`run_local.sh`) that handles virtual environment creation, dependency installation, and service startup.

### 1. Clone the repository
```bash
git clone <repository-url>
cd ppe-detection-synergy-project
```

### 2. Navigate to the app directory
```bash
cd ppe-detection-app
```

### 3. Run the application
Execute the setup script:
```bash
bash run_local.sh
```

This script will:
- Create a Python virtual environment (`venv`).
- Install all necessary dependencies from `requirements.txt`.
- Start the **FastAPI Backend** on port `8000`.
- Start the **React Lite Frontend** on port `3000`.

## Docker Deployment

For containerized deployment:

### 1. Build and run with Docker Compose
```bash
docker-compose up --build
```

This will start:
- **Backend**: http://localhost:8000
- **Web UI**: http://localhost:3000
- **Edge Device UI**: http://localhost:3001

### 2. Access the services
- **Backend API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Web Interface**: [http://localhost:3000/react_frontend.html](http://localhost:3000/react_frontend.html)
- **Edge Device UI**: [http://localhost:3001](http://localhost:3001)

## Mobile App

A Cordova-based Android app is available in the `cordova-app/` directory.

### Building the APK
```bash
cd cordova-app
cordova platform add android
cordova build android
```

### Installation
1. Transfer the generated APK to your Android device
2. Install and configure server URL in app settings
3. Use camera or upload features for PPE detection

## Accessing the App

## Project Structure

```text
ppe-detection-app/
├── app/               # FastAPI backend logic
│   ├── main.py        # API endpoints
│   └── inference.py   # AI model inference
├── ui/                # Frontend assets
│   ├── react_frontend.html
│   └── edge_device_ui.html
├── weights/           # Pre-trained model weights (.pt)
├── data/              # Input data (ignored by git)
├── run_local.sh       # Automated startup script
└── requirements.txt   # Python dependencies
cordova-app/           # Mobile app (Cordova)
├── www/               # Web assets
├── platforms/         # Platform-specific code
└── config.xml         # Cordova configuration
docker-compose.yml     # Container orchestration
build.ps1             # Windows build script
```

## Stopping the Services

To stop both the backend and frontend services, simply press **Ctrl+C** in the terminal where the script is running. The script includes a cleanup routine to terminate the background processes automatically.

## API Usage

The main endpoint for prediction is:
- `POST /predict`: Upload an image or video file to receive detection results.

Example curl request:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

## Development

- **Backend**: Modify `ppe-detection-app/app/` files
- **Web UI**: Edit `ppe-detection-app/ui/` HTML files
- **Mobile App**: Update `cordova-app/www/index.html`
- **Docker**: Adjust `docker-compose.yml` for deployment changes

---
*Created for the IITB-AIMLPractice-Project.*
