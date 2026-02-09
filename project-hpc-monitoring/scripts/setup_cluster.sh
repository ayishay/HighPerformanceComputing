#!/bin/bash
# setup_cluster.sh - Set up a local Kubernetes cluster with GPU support
# Prerequisites: Docker, minikube, kubectl, helm
# Usage: bash scripts/setup_cluster.sh

set -e

echo "========================================="
echo "  HPC Monitoring - Cluster Setup"
echo "========================================="

# 1. Start minikube with GPU support
echo "[1/5] Starting minikube with GPU passthrough..."
minikube start \
    --driver=docker \
    --gpus all \
    --cpus 4 \
    --memory 8192 \
    --disk-size 40g

# 2. Install NVIDIA device plugin (so K8s can see GPUs)
echo "[2/5] Installing NVIDIA device plugin..."
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml \
    2>/dev/null || echo "  (already installed)"

# 3. Verify GPUs are visible
echo "[3/5] Verifying GPU visibility..."
echo "  Waiting for device plugin to register GPUs..."
sleep 10
kubectl get nodes -o json | python3 -c "
import sys, json
data = json.load(sys.stdin)
for node in data['items']:
    name = node['metadata']['name']
    gpus = node['status'].get('capacity', {}).get('nvidia.com/gpu', '0')
    print(f'  Node: {name}, GPUs: {gpus}')
" 2>/dev/null || kubectl get nodes -o wide

# 4. Add Helm repos
echo "[4/5] Adding Helm repositories..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo update

# 5. Create namespace
echo "[5/5] Creating hpc-monitoring namespace..."
kubectl create namespace hpc-monitoring 2>/dev/null || echo "  (already exists)"

echo ""
echo "========================================="
echo "  Cluster setup complete!"
echo "  Next: bash scripts/deploy.sh"
echo "========================================="
