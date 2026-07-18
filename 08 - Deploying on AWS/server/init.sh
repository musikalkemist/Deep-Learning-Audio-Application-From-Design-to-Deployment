#!/bin/bash
# NOTE: deploy.sh is the preferred automation script as it handles disk expansion and swap.

echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y curl bc

# Check current disk size
DISK_SIZE=$(df -h / | awk 'NR==2 {print $2}' | sed 's/G//' | sed 's/M/0.1/')
if (( $(echo "$DISK_SIZE < 10" | bc -l) )); then
    echo "⚠️  WARNING: Disk size is only ${DISK_SIZE}G. TensorFlow builds may fail."
    echo "   Consider using ./deploy.sh to expand your partition."
fi

# install docker
echo "Installing Docker engine..."
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# install latest docker compose
echo "Installing latest Docker Compose v2..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# verify docker is ready and clean cache
echo "Verifying Docker..."
for i in {1..5}; do
    if sudo docker info >/dev/null 2>&1; then
        sudo docker system prune -f
        break
    fi
    sleep 2
done

# build and run docker containers
echo "Building and launching containers..."
cd "$(dirname "$0")"
sudo docker-compose up --build
