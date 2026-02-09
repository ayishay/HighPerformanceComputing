#!/bin/bash
# deploy.sh - Deploy the full monitoring stack to Kubernetes
# Prerequisites: setup_cluster.sh has been run
# Usage: bash scripts/deploy.sh

set -e

NAMESPACE="hpc-monitoring"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "  HPC Monitoring - Deploy Stack"
echo "========================================="

# 1. Build custom Docker images
echo "[1/5] Building custom Docker images..."
eval $(minikube docker-env)

docker build -t hpc-monitoring/gpu-exporter:latest "$PROJECT_DIR"
docker build -t hpc-monitoring/nccl-exporter:latest "$PROJECT_DIR"
docker build -t hpc-monitoring/recommender:latest "$PROJECT_DIR"

echo "  Images built successfully."

# 2. Install Prometheus
echo "[2/5] Installing Prometheus..."
helm upgrade --install prometheus prometheus-community/prometheus \
    --namespace "$NAMESPACE" \
    --set server.persistentVolume.enabled=false \
    --set alertmanager.enabled=false \
    --set pushgateway.enabled=false \
    --wait

# 3. Install Grafana
echo "[3/5] Installing Grafana..."
helm upgrade --install grafana grafana/grafana \
    --namespace "$NAMESPACE" \
    --set adminPassword=admin \
    --set service.type=NodePort \
    --set service.nodePort=30300 \
    --set-file dashboards.default.gpu_monitoring\\.json="$PROJECT_DIR/grafana/dashboards/gpu_monitoring.json" \
    --wait

# 4. Deploy custom components via our Helm chart
echo "[4/5] Deploying HPC monitoring components..."
helm upgrade --install hpc-monitoring "$PROJECT_DIR/helm/hpc-monitoring" \
    --namespace "$NAMESPACE" \
    --wait

# 5. Apply Prometheus config
echo "[5/5] Applying Prometheus scrape config..."
kubectl create configmap prometheus-custom-config \
    --from-file="$PROJECT_DIR/config/prometheus.yml" \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "========================================="
echo "  Deployment complete!"
echo ""
echo "  Access Grafana:"
echo "    URL:      http://$(minikube ip):30300"
echo "    Username: admin"
echo "    Password: admin"
echo ""
echo "  Access Prometheus:"
echo "    kubectl port-forward svc/prometheus-server 9090:80 -n $NAMESPACE"
echo "    URL:      http://localhost:9090"
echo ""
echo "  GPU Exporter metrics:"
echo "    kubectl port-forward svc/gpu-exporter 9090:9090 -n $NAMESPACE"
echo "    URL:      http://localhost:9090/metrics"
echo "========================================="
