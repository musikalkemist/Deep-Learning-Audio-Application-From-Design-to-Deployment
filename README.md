# Deep-Learning-Audio-Application-From-Design-to-Deployment
Code for the "[Deep Learning (Audio) Application: From Design to Deployment](https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp)" series on The Sound of AI YouTube channel.

Learn how to build a simple speech recognition system with TensorFlow and deploy it with Docker as a Flask API on Amazon AWS.

This repository provides a comprehensive guide for building a speech recognition system and deploying it to a production environment. It covers the full lifecycle of an audio ML application: from dataset preparation and model implementation in TensorFlow 2 to deployment using Flask, Docker, uWSGI, NGINX, and Amazon AWS.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat&logo=tensorflow)
![Keras 3](https://img.shields.io/badge/Keras-3.0-red?style=flat&logo=keras)
![Flask](https://img.shields.io/badge/Flask-black?style=flat&logo=flask)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### Note on Versioning
> This v3 release is maintained to ensure compatibility with modern deployment environments and dependency management. While the core logic remains identical to the original course, certain configuration files (Dockerfiles, requirements) are optimized for current best practices. Consequently, the original course version has been deprecated; however, it remains available in the [legacy branch](https://github.com/musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment/tree/legacy) for those wishing to follow the video content exactly.


# Table of Contents
* [Getting Started](#getting-started)
* [Course Structure](#course-structure)
* [Dataset Preparation](#1-dataset-preparation)
* [Model Development](#2-model-development)
* [Deployment Strategy](#3-deployment-strategy)

---
## Getting Started

### 1. Prepare the Environment
Ensure you have the necessary dependencies installed. It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Automated Dataset Initialization
To quickly prepare the dataset and all required configuration files for the entire project, run the provided setup script. This script handles the download, preprocessing (MFCC extraction), and propagation of the `data.json` file to all chapter folders.

```bash
# From the project root, simply run:
chmod +x setup_dataset.sh
./setup_dataset.sh
```

### 3. Running Individual Chapters
After initialization, you can navigate to any chapter directory to run the specific scripts (e.g., training, local server, or Docker orchestration).

```bash
cd "05 - Deploying the Speech Recognition System as a Flask API"
python server.py
```

---
## Course Structure

### 1. Dataset Preparation _(scripts not supposed to be run)_
1. **Course Introduction:** _[Video][1yt] | [Slides][1sl]_
2. **Preparing the Speech Dataset:** _[Video][2yt] | [Code][2cd]_

---
### 2. Model Development
3. **Implementing a Speech Recognition System (TF 2):** _[Video][3yt] | [Code][3cd]_
4. **Making Predictions:** _[Video][4yt] | [Code][4cd]_

---
### 3. Deployment Strategy
> **Note:** Refer to the **[Guide]** links below for detailed deployment instructions for each session.
5. **Deploying as a Flask API:** _[Video][5yt] | [Code][5cd] | [Guide][5md]_
6. **Deploying with uWSGI:** _[Video][6yt] | [Slides][6sl] | [Code][6cd] | [Guide][6md]_
7. **Deploying on Docker with NGINX:** _[Video][7yt] | [Slides][7sl] | [Code][7cd] | [Guide][7md]_
8. **Deploying on Amazon AWS:** _[Video][8yt] | [Code][8cd] | [Guide][8md]_


<!-- Reference links for every chapter:
YouTube videos (#yt), PDF-file slides (#sl) and Jupyter Notebooks (#nb) -->
[1yt]: https://www.youtube.com/watch?v=CA0PQS1Rj_4
[1sl]: <01 - Overview/Deep learning (voice) application_  From design to deployment.pdf>

[2yt]: https://www.youtube.com/watch?v=VPJ2jazh_KI
[2cd]: <02 - Preparing the Dataset/prepare_dataset.py>

[3yt]: https://www.youtube.com/watch?v=INawFGUy-nU
[3cd]: <03 - Implementing a Speech Recognition System in TensorFlow 2/train.py>

[4yt]: https://www.youtube.com/watch?v=cgkUate-BFwA
[4cd]: <04 - Making Predictions with the Speech Recognition System/keyword_spotting_service.py>

[5yt]: https://www.youtube.com/watch?v=1rSNlrEzdL4
[5cd]: <05 - Deploying the Speech Recognition System as a Flask API/server.py>
[5md]: <05 - Deploying the Speech Recognition System as a Flask API/README.md>

[6yt]: https://www.youtube.com/watch?v=7vWuoci8nUk
[6sl]: <06 - Deploying the Speech Recognition System with uWSGI/slides/Deploying the Speech Recognition System with uWSGI.pdf>
[6cd]: <06 - Deploying the Speech Recognition System with uWSGI/code/app.ini>
[6md]: <06 - Deploying the Speech Recognition System with uWSGI/README.md>

[7yt]: https://www.youtube.com/watch?v=nABec-XYxgM
[7sl]: <07 - Deploying the Speech Recognition System on Docker with NGINX/slides/Deploying the Speech Recognition System on Docker with NGINX.pdf>
[7cd]: <07 - Deploying the Speech Recognition System on Docker with NGINX/code/docker-compose.yml>
[7md]: <07 - Deploying the Speech Recognition System on Docker with NGINX/code/README.md>

[8yt]: https://www.youtube.com/watch?v=ceNWWxjtG3U
[8cd]: <08 - Deploying on AWS/local/client.py>
[8md]: <08 - Deploying on AWS/README.md>