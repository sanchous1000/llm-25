#!/bin/bash

echo "Checking Qdrant status..."

if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "Qdrant is already running on http://localhost:6333"
    exit 0
fi

if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running"
    echo ""
    echo "Please start Docker Desktop and try again, or:"
    echo ""
    echo "Option 1: Start Docker Desktop"
    echo "  - Open Docker Desktop application"
    echo "  - Wait for it to start"
    echo "  - Run this script again"
    echo ""
    echo "Option 2: Install Qdrant locally (macOS)"
    echo "  brew install qdrant"
    echo "  qdrant"
    echo ""
    echo "Option 3: Use Qdrant Cloud"
    echo "  - Sign up at https://cloud.qdrant.io"
    echo "  - Update config.yaml with your cloud URL and API key"
    exit 1
fi

if docker ps -a | grep -q qdrant; then
    echo "Starting existing Qdrant container..."
    docker start qdrant
else
    echo "Creating and starting new Qdrant container..."
    docker run -d \
        --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant
fi

echo "Waiting for Qdrant to start..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "Qdrant is running on http://localhost:6333"
        echo "  Dashboard: http://localhost:6333/dashboard"
        exit 0
    fi
    sleep 1
done

echo "Qdrant failed to start"
exit 1

