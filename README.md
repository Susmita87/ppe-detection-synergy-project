# PPE Detection Synergy Project

A real-time Personal Protective Equipment (PPE) detection system using AI (FAST API & React Lite). This application processes images and videos to identify safety gear violations (e.g., missing helmets, vests, etc.).

## 🚀 Key Features

- **Real-time Detection**: Processes images and video streams.
- **Visual Feedback**: Bounding boxes and status overlays on detected people.
- **Safety Status**: Instant "SAFE" or "VIOLATION DETECTED" indicators.
- **Web Interface**: Lightweight React-based UI for uploading and viewing results.

## 🛠 Prerequisites

- Python 3.9 or higher
- Bash (for running the setup script)

## 🏗 Setup & Installation

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

## 🌐 Accessing the App

Once the services are started, you can access them at:

- **Frontend (UI)**: [http://localhost:3000/react_frontend.html](http://localhost:3000/react_frontend.html)
- **Backend (API Health)**: [http://localhost:8000/](http://localhost:8000/)

## 📂 Project Structure

```text
ppe-detection-app/
├── app/               # FastAPI backend logic
│   ├── main.py        # API endpoints
│   └── inference.py   # AI model inference
├── ui/                # Frontend assets
│   └── react_frontend.html
├── weights/           # Pre-trained model weights (.pt)
├── data/              # Input data (ignored by git)
├── run_local.sh       # Automated startup script
└── requirements.txt   # Python dependencies
```

## 🛑 Stopping the Services

To stop both the backend and frontend services, simply press **Ctrl+C** in the terminal where the script is running. The script includes a cleanup routine to terminate the background processes automatically.

## 🧪 API Usage

The main endpoint for prediction is:
- `POST /predict`: Upload an image or video file to receive detection results.

---
*Created for the IITB-AIMLPractice-Project.*
