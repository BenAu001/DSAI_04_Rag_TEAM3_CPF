#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting RAG server at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
