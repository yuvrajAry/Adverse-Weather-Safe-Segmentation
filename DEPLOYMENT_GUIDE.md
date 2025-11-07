# Deployment Guide

This guide will help you deploy the IDDAW project with the backend on Render and frontend on Vercel.

## Prerequisites

1. GitHub account with your code pushed to a repository
2. [Render account](https://render.com) (free tier available)
3. [Vercel account](https://vercel.com) (free tier available)

## Backend Deployment (Render)

### Step 1: Prepare Your Repository

Ensure your code is pushed to GitHub with the following files in the `project` directory:
- `render.yaml` ✓ (created)
- `requirements_backend.txt` ✓ (exists)
- `Procfile` ✓ (created)
- `backend_api.py` ✓ (exists)
- `start_backend.py` ✓ (exists)

### Step 2: Deploy to Render

1. **Sign in to Render**: Go to [render.com](https://render.com) and sign in with GitHub

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your project

3. **Configure the Service**:
   - **Name**: `iddaw-backend` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `project`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements_backend.txt`
   - **Start Command**: `python start_backend.py`

4. **Environment Variables** (Optional):
   - `PYTHON_VERSION`: `3.11.0`
   - Add any custom variables from `.env.example` if needed

5. **Deploy**: Click "Create Web Service"

6. **Wait for Deployment**: Render will build and deploy your app. This may take 5-10 minutes.

7. **Get Your Backend URL**: Once deployed, copy the URL (e.g., `https://iddaw-backend.onrender.com`)

### Important Notes for Backend:

- **Free Tier Limitations**: 
  - Service spins down after 15 minutes of inactivity
  - First request after spin-down will be slow (30-60 seconds)
  - 750 hours/month free runtime
  
- **Model Files**: If you have large model checkpoint files, you may need to:
  - Use Render Disks for persistent storage
  - Download models during build/startup
  - Use cloud storage (S3, Google Cloud Storage)

## Frontend Deployment (Vercel)

### Step 1: Prepare Your Repository

Ensure your code includes the frontend directory with:
- `vercel.json` ✓ (exists)
- `package.json` ✓ (exists)
- `.env.production.example` ✓ (created)

### Step 2: Create Production Environment File

1. In the `frontend` directory, create `.env.production`:
   ```bash
   VITE_API_BASE_URL=https://your-backend-url.onrender.com
   VITE_USE_MOCKS=false
   ```
   Replace `your-backend-url` with your actual Render URL from above.

### Step 3: Deploy to Vercel

1. **Sign in to Vercel**: Go to [vercel.com](https://vercel.com) and sign in with GitHub

2. **Import Project**:
   - Click "Add New" → "Project"
   - Import your GitHub repository

3. **Configure Project**:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build:client`
   - **Output Directory**: `dist/spa`
   - **Install Command**: `npm install`

4. **Environment Variables**:
   - Add `VITE_API_BASE_URL`: Your Render backend URL
   - Add `VITE_USE_MOCKS`: `false`

5. **Deploy**: Click "Deploy"

6. **Wait for Deployment**: Vercel will build and deploy. Usually takes 1-3 minutes.

7. **Get Your Frontend URL**: Once deployed, you'll get a URL like `https://your-app.vercel.app`

### Step 4: Update CORS Settings

After deploying the frontend, update your backend CORS settings:

1. In `project/backend_api.py`, update the CORS configuration if needed:
   ```python
   CORS(app, resources={
       r"/api/*": {
           "origins": ["https://your-app.vercel.app"],  # Add your Vercel URL
           "supports_credentials": True
       }
   })
   ```

2. Redeploy the backend on Render

## Post-Deployment Checklist

- [ ] Backend health check works: `https://your-backend.onrender.com/health`
- [ ] Frontend loads successfully
- [ ] Frontend can communicate with backend
- [ ] CORS is configured correctly
- [ ] Environment variables are set properly
- [ ] Test the inference endpoint with sample images

## Continuous Deployment

Both Render and Vercel support automatic deployments:

- **Render**: Redeploys automatically when you push to your connected branch
- **Vercel**: Redeploys automatically on every git push

To disable auto-deploy, configure it in the respective dashboard settings.

## Troubleshooting

### Backend Issues

1. **Service won't start**:
   - Check build logs in Render dashboard
   - Verify `requirements_backend.txt` dependencies
   - Ensure Python version is compatible (3.8+)
   - Check if model checkpoint files exist in `ckpts/` folder

2. **Memory issues**:
   - PyTorch models are memory-intensive
   - Consider upgrading to paid tier for more RAM
   - Or optimize model loading

3. **Timeout errors**:
   - Free tier has 30-second timeout
   - Consider upgrading for longer requests
   - Or optimize inference time

### Frontend Issues

1. **Build fails**:
   - Check Vercel build logs
   - Verify all dependencies are in `package.json`
   - Check TypeScript errors

2. **API calls fail**:
   - Verify `VITE_API_BASE_URL` is set correctly
   - Check browser console for CORS errors
   - Ensure backend is running

3. **Environment variables not working**:
   - Vercel requires rebuild after env var changes
   - Ensure variables start with `VITE_` for client-side access

## Monitoring

- **Render**: Check logs and metrics in Render dashboard
- **Vercel**: Check deployment logs and analytics in Vercel dashboard

## Cost Optimization

### Free Tier Limits

**Render**:
- 750 hours/month
- Service spins down after 15 min inactivity
- 512 MB RAM

**Vercel**:
- 100 GB bandwidth/month
- Unlimited deployments
- Serverless function execution time limits

### Tips:
- Use Render's "Suspend" feature when not actively using
- Optimize model size and inference time
- Cache static assets
- Consider CDN for images

## Support

For issues specific to:
- **Render**: [Render Documentation](https://render.com/docs)
- **Vercel**: [Vercel Documentation](https://vercel.com/docs)

For project-specific issues, check the project README and documentation.
