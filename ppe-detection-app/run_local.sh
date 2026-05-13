#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Checking dependencies (this may take a few seconds)..."
# Only install if requirements.txt is newer than the venv or first run
if [ ! -f venv/.pip-lock ] || [ requirements.txt -nt venv/.pip-lock ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.pip-lock
    echo "Dependencies updated."
else
    echo "Dependencies are up to date."
fi

# Cleanup existing processes on ports 8000 and 3000
echo "Clearing ports 8000 and 3000..."
lsof -ti :8000,3000 | xargs kill -9 2>/dev/null || true

# Create temp uploads directory if not exists
mkdir -p temp_uploads

echo "Starting Backend (FastAPI)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Frontend (Nginx/Python)..."
(cd ui && python3 -m http.server 3000) &
REACT_PID=$!

function cleanup {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $REACT_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "Services started!"
echo "- API: http://localhost:8000"
echo "- UI:  http://localhost:3000/react_frontend.html"
echo "Press Ctrl+C only when you want to SHUT DOWN the app."

wait
