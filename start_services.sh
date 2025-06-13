#!/bin/bash

echo "Starting Ollama service..."

python3 -m ad_updater.main &

# Start Ollama in the background
ollama serve &

wait -n