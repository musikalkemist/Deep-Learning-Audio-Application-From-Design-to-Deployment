## Troubleshooting Guide: Deploying Deep Learning Apps on AWS EC2 Free Tier
Deploying a machine learning application (like a TensorFlow/Flask audio processor) on an AWS EC2 Free Tier instance (`t3.micro`) is a great way to host a project, but you will quickly hit the physical limits of the server.

This guide covers the four most common roadblocks you will encounter—outdated tools, lack of disk space, unallocated partitions, and memory crashes—and provides the exact terminal commands to fix them.

---
### Issue 0: Docker is not installed or the service is down
If you see an error like `failed to connect to the docker API at unix:///var/run/docker.sock` or `docker-compose: command not found`, it means the Docker engine or the Compose tool is missing from your system. Standard Ubuntu AMIs do not come with Docker pre-installed.

**The Fix: Install and start the Docker engine**

1. Update your system and install the Docker package:
```bash
sudo apt-get update
sudo apt-get install -y docker.io
```

2. Ensure the Docker service is active and set to start on boot:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

3. (Optional) If you don't want to use `sudo` every time, add your user to the docker group (requires logout/login to take effect):
```bash
sudo usermod -aG docker $USER
```

---
### Issue 1: Docker Compose Fails Due to an Outdated Version
Sometimes, Ubuntu's default package manager installs a severely outdated version of Docker Compose (e.g., v1.25) that conflicts with modern `docker-compose.yml` files. You need to force the system to use the latest version.

**The Fix: Aggressively reinstall Docker Compose**

1. Force Ubuntu to uninstall the broken package:
```bash
sudo apt-get remove docker-compose -y
```

2. Hunt down and delete any lingering old binaries:
```bash
sudo rm -f /usr/bin/docker-compose
sudo rm -f /usr/local/bin/docker-compose
```

3. Download the brand-new version directly into your main system folder:
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/bin/docker-compose
```

4. Give it permission to run:
```bash
sudo chmod +x /usr/bin/docker-compose
```

5. Verify the installation worked (it should print `v2.x.x`):
```bash
docker-compose --version
```

---
### Issue 2: Build Fails with `No space left on device`
By default, an AWS EC2 instance only provisions an **8GB** hard drive. Heavy libraries like `scipy`, `ffmpeg`, and `tensorflow` combined with Docker's build cache will rapidly eat up this space, causing the build to crash.

**The Fix: Clear Cache & Expand AWS Storage**

1. **Clear the Docker Cache:** Remove dangling images and half-finished builds to reclaim immediate space:
```bash
sudo docker system prune -a --volumes -f
sudo docker builder prune -a -f
```

2. **Upgrade your AWS Volume:** If 8GB is simply too small for your app, you can increase your limit up to **30GB** for free under the AWS Free Tier. **(⚠️ CAUTION: Additional costs could be incurred)**
* Go to the **AWS EC2 Console**.
* Click **Instances**, select your server, and click the **Storage** tab.
* Click the **Volume ID** (`vol-...`).
* Select the volume, click **Actions > Modify volume**.
* Change the size to `20` or `30` (GB) and save.

> NOTE: Read the [Configuration Guide](SETUP.md) for further details.

---
### Issue 3: Ubuntu Doesn't Recognize the New Disk Space
If you expanded your disk in the AWS Console to 30GB but running `df -h` still shows the `/dev/root` partition at 98% capacity on an 8GB drive, the operating system doesn't know about the new space yet. You must manually resize the Linux partition.

**The Fix: Force Ubuntu to expand into the new space**

1. Expand the main partition to fill the whole physical disk (pay attention to the space before the `1`):
```bash
sudo growpart /dev/nvme0n1 1
```

2. Expand the Ubuntu filesystem to recognize the newly freed space:
```bash
sudo resize2fs /dev/root
```

3. Verify that your root directory now shows the full 20GB or 30GB:
```bash
df -h
```

---
### Issue 4: `504 Gateway Time-out` (Server runs out of RAM)
If your app successfully builds and Nginx catches your requests, but you receive a `504 Gateway Time-out` after sending an audio file, your server is likely running out of memory.

A `t3.micro` instance only has **1GB of RAM**. When Flask attempts to load TensorFlow and process audio, the memory hits 100%, the server freezes, and Nginx times out the connection (usually after 60 seconds).

**The Fix: Create a Swap File (Fake RAM)**
You can use a portion of your newly expanded hard drive to act as overflow RAM (Swap memory). This is slower than physical RAM but prevents the server from crashing.

1. Allocate a 2GB file on your hard drive:
```bash
sudo fallocate -l 2G /swapfile
```

2. Lock down the file permissions (critical for security):
```bash
sudo chmod 600 /swapfile
```

3. Format the file to act as Swap memory:
```bash
sudo mkswap /swapfile
```

4. Turn the Swap memory on:
```bash
sudo swapon /swapfile
```

5. Verify that your system now has 2GB of Swap available:
```bash
free -h
```

> **Note:** Because Swap memory relies on the hard drive, your *first* API request that triggers the Deep Learning model may take 15-30 seconds to process. Subsequent requests will be much faster.

---
### Issue 5: How to completely reset the environment for a clean update
If you want to ensure you are starting from a "blank slate" when uploading new code, or if you keep hitting disk space errors despite expanding your drive, you should wipe the existing environment.

**The Fix: Stop, Delete, and Deep Clean**

1. **Stop and Wipe Current Containers:** This stops the app and deletes the specific images and volumes created for this project.
```bash
cd ~/server
sudo docker-compose down --rmi all -v
```

2. **Delete the Project Folder:** This removes the actual code files from your home directory.
```bash
cd ~
rm -rf server
```

3. **Deep Clean Docker (Crucial for Space):** This reclaims every possible byte from the Docker cache, including base images (like TensorFlow).
```bash
sudo docker system prune -a --volumes -f
```

4. **Re-upload and Deploy:** You can now upload your fresh `server/` folder from your local machine and run `./deploy.sh` for a clean install.