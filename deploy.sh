#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e


echo "🧹 Stopping and removing old containers..."
docker compose down --remove-orphans

echo "🚀 Building Docker images..."
docker compose build

echo "🚀 Starting containers in detached mode..."
docker compose up -d

echo "⏳ Waiting for FastAPI app to start..."

# Optional: wait for a few seconds to allow startup
sleep 5

echo "✅ FastAPI app is running at http://localhost:8000"
