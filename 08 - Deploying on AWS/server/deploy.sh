#!/bin/bash

# Exit on any error
set -e

echo "--------------------------------------------------------"
echo "Starting Deep Learning Audio App Deployment Automation..."
echo "--------------------------------------------------------"

# 1. Update system packages and Install dependencies
echo "Updating system packages and installing tools..."
sudo apt-get update -y
sudo apt-get install -y curl cloud-guest-utils bc

# 2. Expand Partition (Critical for 8GB -> 16GB/30GB upgrades)
echo "Expanding root partition to utilize full disk space..."
# Check current disk size to provide a helpful warning
DISK_SIZE=$(df -h / | awk 'NR==2 {print $2}' | sed 's/G//' | sed 's/M/0.1/')
if (( $(echo "$DISK_SIZE < 10" | bc -l) )); then
    echo "⚠️  WARNING: Your disk size is only ${DISK_SIZE}G."
    echo "   TensorFlow requires more space. If this step fails, ensure you have"
    echo "   increased the Volume Size to 16GB or 30GB in the AWS Console."
fi

# Detect root device and partition number automatically
# This works for /dev/nvme0n1p1, /dev/xvda1, etc.
ROOT_DEV_FULL=$(findmnt / -nNo SOURCE)
ROOT_DEV_NAME=$(lsblk -no pkname "$ROOT_DEV_FULL")
ROOT_PART_NUM=$(echo "$ROOT_DEV_FULL" | grep -o '[0-9]*$')

echo "Detected root device: /dev/$ROOT_DEV_NAME (partition $ROOT_PART_NUM)"

# Expand the main partition
sudo growpart "/dev/$ROOT_DEV_NAME" "$ROOT_PART_NUM" || echo "Partition already expanded or no unallocated space found."

# Expand the Ubuntu filesystem
sudo resize2fs "$ROOT_DEV_FULL" || echo "Filesystem resize skipped or not required."

# Verify disk expansion
echo "Current disk status:"
df -h /

# 3. Allocate Swap Memory (Prevents RAM crashes during model loading)
if [ ! -f /swapfile ]; then
    echo "Allocating 2GB of Swap Memory (Fake RAM)..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo "Swap memory enabled."
else
    echo "Swap file already exists. Skipping allocation."
fi

# 4. Clean up and Install Docker
echo "Installing Docker engine..."
# Remove potential conflicts or broken installs
sudo apt-get remove -y docker-compose docker-compose-v2 docker.io containerd runc || true

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# 5. Install latest Docker Compose v2
echo "Installing latest Docker Compose v2..."
sudo rm -f /usr/bin/docker-compose /usr/local/bin/docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
# Link for compatibility
sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
echo "Docker Compose version: $(docker-compose --version)"

# 6. Wait for Docker Daemon and Clean Cache
echo "Waiting for Docker daemon to start..."
for i in {1..15}; do
    if sudo docker info >/dev/null 2>&1; then
        echo "Docker is ready! Cleaning up space..."
        # Reclaim space from failed builds or old images
        sudo docker system prune -f
        break
    fi
    if [ $i -eq 15 ]; then
        echo "ERROR: Docker daemon failed to start. Checking status..."
        sudo systemctl status docker
        exit 1
    fi
    sleep 2
done

# 7. Build and Run Containers
echo "Building and launching Docker containers (this might take a few minutes)..."
# Ensure we are in the server directory
cd "$(dirname "$0")"
sudo docker-compose up -d --build

echo "--------------------------------------------------------"
echo "Deployment Complete!"
echo "Your API is now running on port 80."
echo "You can test it using the client.py script in the local/ folder."
echo "--------------------------------------------------------"
