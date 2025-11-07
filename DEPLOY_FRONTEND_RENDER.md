# 🚀 Deploy Frontend to Render - Complete Guide

Yes! You can deploy your frontend on Render as a **Static Site**. This is actually simpler than Vercel and keeps everything in one place!

## ✅ Prerequisites
- ✅ Backend deployed on Render: `https://adverse-weather-safe-segmentation.onrender.com`
- ✅ GitHub repository: `https://github.com/yuvrajAry/Adverse-Weather-Safe-Segmentation`
- ✅ Render account (you already have one!)

---

## 📋 Option 1: Using render.yaml (Recommended - Auto Setup)

Your `render.yaml` has been updated to include both backend and frontend services!

### Step 1: Push to GitHub
The `render.yaml` file in the `project` directory has been updated. Push it:
```bash
git add project/render.yaml
git commit -m "Add frontend static site to render.yaml"
git push origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) → Dashboard
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Select **"Adverse-Weather-Safe-Segmentation"**
5. Render will detect `render.yaml` and create both services automatically!

### Step 3: Verify Services
You should see:
- ✅ **iddaw-backend** (Web Service - Python)
- ✅ **iddaw-frontend** (Static Site)

Both will deploy automatically!

---

## 📋 Option 2: Manual Setup (If Blueprint doesn't work)

### Step 1: Create Static Site
1. Go to [render.com](https://render.com) → Dashboard
2. Click **"New +"** → **"Static Site"**
3. Connect your GitHub repository
4. Select **"Adverse-Weather-Safe-Segmentation"**

### Step 2: Configure Static Site

**Basic Settings:**
- **Name**: `iddaw-frontend`
- **Region**: Oregon (or closest to you)
- **Branch**: `main`
- **Root Directory**: `frontend` ⚠️ **IMPORTANT**

**Build Settings:**
- **Build Command**: `npm install && npm run build:client`
- **Publish Directory**: `dist/spa`

**Environment Variables:**
- `VITE_API_BASE_URL` = `https://adverse-weather-safe-segmentation.onrender.com`
- `VITE_USE_MOCKS` = `false`

### Step 3: Deploy
Click **"Create Static Site"** and wait for deployment (2-5 minutes)

---

## ✅ After Deployment

### Your URLs:
- **Backend**: `https://adverse-weather-safe-segmentation.onrender.com`
- **Frontend**: `https://iddaw-frontend.onrender.com` (or similar)

### Test Your Deployment:
1. Visit your frontend URL
2. Open browser console (F12)
3. Check that API calls go to your backend
4. Try logging in and uploading images

---

## 🔄 Auto-Deploy

Both services will automatically redeploy when you push to GitHub `main` branch!

---

## 💡 Advantages of Render for Frontend

✅ **Everything in one place** - Backend and frontend on same platform
✅ **Simpler setup** - No need for separate Vercel account
✅ **Unified dashboard** - Manage both services together
✅ **Same free tier** - 750 hours/month for static sites too
✅ **No build issues** - Render handles Vite builds well

---

## ⚠️ Important Notes

### Free Tier Limitations:
- **Static sites**: Unlimited bandwidth on free tier
- **Backend**: 750 hours/month, spins down after 15 min inactivity
- **First request**: Might be slow if backend spun down (~30-60s)

### If You Get CORS Errors:
Your backend already allows all origins (`*`), so CORS should work. If not, update `project/backend_api.py`:
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["*", "https://iddaw-frontend.onrender.com"],
        "supports_credentials": True
    }
})
```

---

## 🆘 Troubleshooting

### Build Fails:
1. Check build logs in Render dashboard
2. Verify `rootDir` is set to `frontend`
3. Verify `buildCommand` is correct
4. Check that `dist/spa` directory is created after build

### Frontend Can't Connect to Backend:
1. Verify `VITE_API_BASE_URL` environment variable is set correctly
2. Check browser console for API errors
3. Test backend separately: `https://adverse-weather-safe-segmentation.onrender.com/api/ping`

### Static Site Not Loading:
1. Verify `Publish Directory` is `dist/spa`
2. Check that build completed successfully
3. Clear browser cache and try again

---

## 📊 Deployment Status

### Backend (Render) ✅
- **URL**: https://adverse-weather-safe-segmentation.onrender.com
- **Status**: Live and tested

### Frontend (Render) 🔄
- **Status**: Ready to deploy
- **Configuration**: Complete in render.yaml

---

**Good luck! Your frontend will be live on Render in just a few minutes! 🎉**

