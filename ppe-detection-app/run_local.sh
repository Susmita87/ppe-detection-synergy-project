#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Cleanup existing processes on ports 8000 and 3000
echo "Clearing ports 8000 and 3000..."
lsof -ti :8000,3000 | xargs kill -9 2>/dev/null

# Create temp uploads directory if not exists
mkdir -p temp_uploads

echo "Starting Backend (FastAPI) on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Lite React Frontend on port 3000..."
# 🔹 Use Python's built-in server to host the HTML
(cd ui && python3 -m http.server 3000) &
REACT_PID=$!

function cleanup {
    echo "Stopping services..."
    kill $BACKEND_PID
    kill $REACT_PID
}

trap cleanup EXIT

echo "Services started!"
echo "- API: http://localhost:8000"
echo "- UI (React Lite): http://localhost:3000/react_frontend.html"
echo "Press Ctrl+C to stop both."

wait
