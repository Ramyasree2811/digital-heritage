# Cloud Deployment Guide — Digital Heritage Restoration Platform

This guide provides step-by-step instructions to deploy the Digital Heritage Restoration platform to cloud providers.

---

## Architecture Overview

- **Web Framework:** Flask with Gunicorn WSGI
- **Deep Learning Runtime:** PyTorch (CPU-optimized for cost-effective cloud hosting; GPU supported when NVIDIA runtime is enabled)
- **OCR Engine:** EasyOCR (Hindi, Sanskrit, English)
- **Containerization:** Docker (`python:3.11-slim`)
- **Port:** Configured via `PORT` environment variable (default: `5000`)

---

## Method 1: Hugging Face Spaces (Recommended for Research & Demos)
*Completely free, high bandwidth, built-in SSL, permanent public URL.*

1. **Create Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces).
   - Click **Create new Space**.
   - Select **Docker** as the Space SDK and choose **Blank**.
   - Space Hardware: Free (2 vCPU, 16 GB RAM).

2. **Push Code to Space:**
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/digital-heritage-restorer
   git push space main
   ```
3. **Automatic Build:**
   - Hugging Face automatically detects the [Dockerfile](file:///c:/Users/DELL/OneDrive/Desktop/digital/Dockerfile), installs dependencies, and launches the live web app on port 7860/5000.

---

## Method 2: Render.com (Easiest Automated Git Push)
*Free starter tier, automated continuous deployment on GitHub push.*

1. **Push your repository to GitHub:**
   ```bash
   git add .
   git commit -m "Production release: Heritage Restoration Engine"
   git push origin main
   ```
2. **Deploy on Render:**
   - Sign up / Log in to [render.com](https://render.com).
   - Click **New +** → **Web Service**.
   - Connect your GitHub repository.
   - Choose **Docker** as the Runtime.
   - Set **Instance Type**: Starter or Standard (1 GB - 2 GB RAM recommended).
   - Click **Create Web Service**.
3. **Live Access:**
   - Render builds the container and provides a public HTTPS URL (e.g., `https://digital-heritage-restoration.onrender.com`).

---

## Method 3: Google Cloud Run (Serverless Container with Auto-Scale)
*Fast, scales to zero when idle, low cost.*

1. **Install and authenticate Google Cloud SDK:**
   ```bash
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   ```
2. **Build and push image to Google Artifact Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/heritage-restorer:latest
   ```
3. **Deploy container:**
   ```bash
   gcloud run deploy heritage-restorer \
       --image gcr.io/<YOUR_PROJECT_ID>/heritage-restorer:latest \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --memory 2Gi \
       --cpu 2 \
       --timeout 300
   ```
4. Cloud Run will display your live Service URL.

---

## Method 4: Amazon Web Services (AWS App Runner or EC2)

### Option A: AWS App Runner
1. Authenticate with Amazon Elastic Container Registry (ECR):
   ```bash
   aws ecr create-repository --repository-name heritage-restorer
   docker tag digital_heritage_restorer:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/heritage-restorer:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/heritage-restorer:latest
   ```
2. Open AWS App Runner Console → **Create Service** → Select ECR image → Set Port `5000` → Deploy.

### Option B: Ubuntu EC2 Instance
1. Launch an Ubuntu 22.04 LTS instance (t3.medium or g4dn.xlarge for GPU).
2. SSH into instance:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose git
   git clone <YOUR_REPO_URL> && cd digital
   sudo docker-compose up -d --build
   ```
3. Open inbound security group port `5000` in the AWS console.

---

## Method 5: Local Docker Container Execution

To test the containerized service on your local machine:
```bash
# Build Docker image
docker build -t heritage-restorer .

# Run container
docker run -p 5000:5000 heritage-restorer
```
Open your browser at `http://localhost:5000`.
