# Deploying AW-SafeSeg with Docker

This document shows how to run the backend locally with Docker and how to publish an image to Docker Hub using the included GitHub Actions workflow.

## Local: Run with docker-compose

1. Build and start the backend (mounts `project/ckpts` so your checkpoints are available inside the container):

```powershell
cd d:\iddaw\pro
docker-compose up --build -d
```

2. Check logs:

```powershell
docker-compose logs -f backend
```

3. Stop:

```powershell
docker-compose down
```

Notes:
- The container exposes port 8000; the frontend (if run separately) should point its API base to `http://localhost:8000`.
- Make sure there are valid checkpoint files in `project/ckpts/` or the model will print a warning and the predict endpoint will return errors.

## Publish image to Docker Hub (CI)

1. Create Docker Hub account and a repository (e.g. `yuvrajAry/adverse-weather-backend`).
2. Add secrets in GitHub repository settings:
   - `DOCKERHUB_USERNAME` - your Docker Hub username
   - `DOCKERHUB_TOKEN` - a Docker Hub access token (or password)
3. Push to `main` branch. The workflow `.github/workflows/docker-publish.yml` builds and pushes the image using the repository secrets.

## Deploy to a cloud provider

Common options:
- Google Cloud Run: deploy the pushed image from Docker Hub or GitHub Container Registry. You can use `gcloud run deploy` or set up a GitHub Action for Cloud Run and provide `GCP_SERVICE_ACCOUNT` as a secret.
- Render / Railway: these providers can directly build from your repo or from the pushed Docker image.
- AWS ECS / Fargate: push image to ECR and create a service.

Important: model checkpoints can be large. For production, consider hosting checkpoints in cloud storage (S3 / GCS) and download them at container startup or mount them in the container at deploy time.

## Frontend

Recommended: Deploy the React frontend to Netlify or Vercel. Configure `VITE_API_BASE_URL` in the deployment environment to point to your backend.

## Next steps / hardening

- Add health-check endpoint and readiness probe
- Add persistent storage for results (S3) and use a managed database for production (Postgres)
- Use HTTPS (TLS) and configure CORS properly
- Add secret management and rotate keys

