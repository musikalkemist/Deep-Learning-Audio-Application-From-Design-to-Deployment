# Speech Recognition System - AWS Deployment
> UPDATED: 06/08/2026

This directory contains the necessary components to deploy the keyword spotting system to an AWS EC2 instance using Docker and NGINX.

## Directory Structure
- `local/`:
    - `classifier/`: Scripts for local preprocessing and training.
    - `test/`: Sample audio files for testing.
    - `client.py`: Python client to test the API (requires updating the server IP).
- `server/`:
    - `flask/`: Flask application code (`server.py`), uWSGI config, and the trained model.
    - `nginx/`: NGINX configuration for reverse proxying.
    - `docker-compose.yml`: Orchestrates the Flask and NGINX services.
    - `init.sh`: Shell script to automate the setup of the AWS EC2 instance.
    - `deploy.sh`: (OPTIONAL) Script that automates Docker installation, partition expansion, and memory management.
- `speech_recognition.pem`: Private key for SSH access to the AWS instance (Note: Download your key here and Keep this secure!).

## Deployment Workflow
### 1. Setting up the AWS EC2 Instance (Recommended Specs)
To ensure the Deep Learning model runs without crashing, use these settings:
1.  **AMI:** Ubuntu Server 26.04 LTS (Free tier eligible).
2.  **Instance Type:** `t3.micro` (Provides better performance than t2.micro).
3.  **Storage:** Increase the Root Volume size to **16 GB** (or up to 30 GB for free).
4.  **Security Group:** Allow inbound traffic on:
    - **SSH (Port 22):** For management.
    - **HTTP (Port 80):** For the NGINX proxy (which forwards to port 1234 internally).

> NOTE: See the [Configuration Guide](SETUP.md) for more details.

**AWS Configuration Checklist:**
* [ ] **AMI & Compute:** Launch `t3.micro` using Ubuntu Server 26.04 LTS.
* [ ] **Key Pair:** Download the `.pem` file and save it securely.
* [ ] **Networking & Security:** Open Port 80 (HTTP) for the NGINX proxy traffic.
* [ ] ❗️**Clean-Up (Post-Testing):** **Terminate** the instance when finished; stopping it will not halt EBS storage billing.

### 2. Transferring Files to AWS
From your local machine, upload the `server/` directory to your instance:
```bash
scp -i "speech_recognition.pem" -r server/ ubuntu@<AWS-INSTANCE-PUBLIC-DNS>:~/
```

### 3. Initializing the Server
Connect to your instance via SSH to begin the configuration:
```bash
ssh -i "speech_recognition.pem" ubuntu@<AWS-INSTANCE-PUBLIC-DNS>
```

Choose one of the following deployment strategies based on your infrastructure needs.

#### ***Option A: Standard Deployment***
Use this option if your instance already has sufficient storage and memory allocated. Navigate to the `server` folder and execute the initialization script:

```bash
cd server
chmod +x init.sh
./init.sh
```

This process builds the environment and installs dependencies. It may take several minutes to complete depending on the size of the packages being downloaded.

> **NOTE:** If you encounter any issues during the build process, refer to the [Troubleshooting Guide](https://www.google.com/search?q=TROUBLESHOOTING.md).

#### ***Option B: Deployment with Resource Scaling***
> **⚠️ CAUTION: Additional costs could be incurred if you exceed the free tier limits.**

If you are using an EC2 free-tier machine (`t3.micro` with 1GB RAM), the default resources are typically insufficient for heavy machine learning workloads and can lead to Out-Of-Memory (OOM) errors.

> **NOTE:** New users receive 6 months of free tier credits, which generally cover minor upscaling costs.

1. Inside the AWS panel, increase the volume size to **16 GB or more** (see this [guide](SETUP.md))

2. Instead of running `init.sh`, execute the automation script to dynamically scale your system resources:

```bash
cd server
chmod +x deploy.sh
./deploy.sh
```

This script handles the heavy lifting of environment preparation:

* Installs the **Docker Engine** and the latest **Docker Compose v2**.
* **Expands the disk partition** to ensure the OS recognizes the full 16 GB/30 GB volume size.
* Allocates **2 GB of Swap Memory** (using disk space as overflow RAM) to prevent heavy frameworks like TensorFlow or PyTorch from crashing during model loading.
* Builds and starts your API containers.

### 4. Testing the Remote API
1.  On your **local machine**, open [`local/client.py`](local/client.py).
2.  Update the `IP_ADDRESS` variable with your EC2 instance's public IP:
    ```python
    IP_ADDRESS = "3.21.233.224" # Replace with your IP
    ```
3.  Run the client:
    ```bash
    python local/client.py
    ```
## Helpful Resources
- **[AWS EC2 Configuration Guide (SETUP.md)](SETUP.md)**: Detailed steps for provisioning your instance with screenshots.
- **[Troubleshooting Guide (TROUBLESHOOTING.md)](TROUBLESHOOTING.md)**: Fixes for common errors like "No space left on device" and "504 Gateway Timeout".

## Security Note
- ⚠️ **CAUTION:** Storage allocation costs depend entirely on your AWS account status and current tier policies. Ensure your account accommodates this configuration to avoid unexpected charges.
- **Private Key:** Never share your `.pem` file or commit it to a public repository.
- **Firewall:** Only open necessary ports in your AWS Security Group.