# Quick Deployment Steps

## 🚀 Deploy Backend to Render (5 minutes)

1. **Push to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com) → Sign in with GitHub
   - Click "New +" → "Web Service"
   - Select your repository
   - Configure:
     - **Name**: `iddaw-backend`
     - **Root Directory**: `project`
     - **Build Command**: `pip install -r requirements_backend.txt`
     - **Start Command**: `python start_backend.py`
   - Click "Create Web Service"
   - **Copy the deployed URL** (e.g., `https://iddaw-backend.onrender.com`)

## 🎨 Deploy Frontend to Vercel (3 minutes)

1. **Create `.env.production`** in the `frontend` folder:
   ```bash
   VITE_API_BASE_URL=https://your-backend-url.onrender.com
   VITE_USE_MOCKS=false
   ```
   ⚠️ Replace `your-backend-url` with your actual Render URL

2. **Deploy on Vercel**:
   - Go to [vercel.com](https://vercel.com) → Sign in with GitHub
   - Click "Add New" → "Project"
   - Import your repository
   - Configure:
     - **Root Directory**: `frontend`
     - **Framework Preset**: Vite
     - **Build Command**: `npm run build:client`
     - **Output Directory**: `dist/spa`
   - Add Environment Variables:
     - `VITE_API_BASE_URL`: Your Render backend URL
     - `VITE_USE_MOCKS`: `false`
   - Click "Deploy"

## ✅ Test Your Deployment

1. **Backend Health Check**: Visit `https://your-backend.onrender.com/health`
   - Should return: `{"status": "ok"}`

2. **Frontend**: Visit your Vercel URL
   - Should load the application
   - Try uploading RGB and NIR images

## 🔄 Update CORS (If Needed)

If you get CORS errors, update `project/backend_api.py` (find the CORS section and add your Vercel URL):

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["*", "https://your-app.vercel.app"],
        "supports_credentials": True
    }
})
```

Then push to GitHub - Render will auto-redeploy.

## 📝 Notes

- **First Request**: Backend may take 30-60 seconds on first request (free tier cold start)
- **Auto-Deploy**: Both services auto-deploy on git push
- **Logs**: Check deployment logs in respective dashboards for issues

## 🆘 Need Help?

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions and troubleshooting.
