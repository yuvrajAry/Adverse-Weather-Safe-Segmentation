# 🚀 Deploy Frontend to Vercel - Complete Guide

## ✅ Prerequisites
- ✅ Backend deployed on Render: `https://adverse-weather-safe-segmentation.onrender.com`
- ✅ GitHub repository: `https://github.com/yuvrajAry/Adverse-Weather-Safe-Segmentation`
- ✅ Vercel account (sign up at [vercel.com](https://vercel.com) with GitHub)

---

## 📋 Step-by-Step Deployment

### Step 1: Sign in to Vercel
1. Go to [https://vercel.com](https://vercel.com)
2. Click **"Sign up"** or **"Login"**
3. Choose **"Continue with GitHub"**
4. Authorize Vercel to access your repositories

---

### Step 2: Create New Project
1. On your Vercel dashboard, click **"Add New..."** (top right)
2. Select **"Project"**
3. You'll see a list of your GitHub repositories
4. Find and click **"Import"** next to **"Adverse-Weather-Safe-Segmentation"**

---

### Step 3: Configure Project Settings

#### A. Framework Preset
- Vercel should auto-detect **"Vite"** ✅
- If not, manually select **"Vite"** from the dropdown

#### B. Root Directory ⚠️ **CRITICAL**
- Click **"Edit"** next to "Root Directory"
- Type: `frontend`
- Click **"Continue"**

#### C. Build Settings (verify these match):
```
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build:client
Output Directory: dist/spa
Install Command: npm install
```

---

### Step 4: Environment Variables ⚠️ **REQUIRED**

Click **"Environment Variables"** section to expand it.

Add **TWO** variables:

**Variable 1:**
- **Name**: `VITE_API_BASE_URL`
- **Value**: `https://adverse-weather-safe-segmentation.onrender.com`
- **Environment**: Select all (Production, Preview, Development)
- Click **"Add"**

**Variable 2:**
- **Name**: `VITE_USE_MOCKS`
- **Value**: `false`
- **Environment**: Select all (Production, Preview, Development)
- Click **"Add"**

**Important Notes:**
- ⚠️ Do NOT include trailing slash in the URL
- ⚠️ Make sure both variables are added
- ⚠️ Select all environments for both variables

---

### Step 5: Deploy!
1. Review all settings one more time
2. Click the big blue **"Deploy"** button at the bottom
3. Vercel will start building your project
4. Watch the build logs (optional but helpful!)

---

## ⏱️ Build Process (2-5 minutes)

You'll see logs like:
```
Cloning repository...
Installing dependencies...
Running build command...
npm run build:client
Building...
✓ built in 45s
Uploading...
Deployment complete!
```

---

## ✅ After Successful Deployment

You'll see:
- ✅ **Green checkmark**
- **"Visit"** button
- Your app URL (e.g., `https://adverse-weather-safe-segmentation.vercel.app`)

**Click "Visit"** to open your app!

---

## 🧪 Testing Your Deployed App

### 1. Open Your App URL
The URL will be something like:
- `https://adverse-weather-safe-segmentation.vercel.app` or
- `https://adverse-weather-safe-segmentation-[username].vercel.app`

### 2. Check Browser Console
1. Open your app in browser
2. Press `F12` to open Developer Tools
3. Go to **"Console"** tab
4. You should see:
```
API Config: {
  baseURL: "https://adverse-weather-safe-segmentation.onrender.com",
  USE_MOCKS: false
}
```

### 3. Test Authentication
1. Click **"Sign Up"** or **"Login"**
2. Create a new account or use test credentials:
   - Email: `test@example.com`
   - Password: `password123`
3. You should be able to log in successfully

### 4. Test Image Upload & Prediction
1. After logging in, go to the upload page
2. Upload RGB and NIR images
3. Wait for prediction to complete
4. View segmentation results with:
   - Segmentation masks
   - Confidence heatmaps
   - Safety analysis overlays

---

## 🔗 How It Works

### Connection Flow:
1. **Frontend (Vercel)** → User interacts with React app
2. **API Calls** → Frontend makes requests to `VITE_API_BASE_URL`
3. **Backend (Render)** → Processes requests and returns results
4. **Response** → Frontend displays results to user

### Environment Variable Magic:
- `VITE_API_BASE_URL` is injected at **build time** by Vercel
- All API calls use this URL automatically
- No code changes needed after deployment!

---

## ❌ Common Errors & Solutions

### Error: "Build failed"
**Possible causes:**
- Missing dependencies
- TypeScript errors
- Build configuration issues

**Solution:**
1. Check build logs in Vercel dashboard
2. Look for specific error messages
3. Fix issues and push to GitHub (auto-redeploys)

### Error: "CORS error" in browser console
**Cause**: Backend not allowing frontend domain

**Solution**: ✅ Already fixed! Backend allows all origins (`*`)

### Error: "API calls going to localhost:8001"
**Cause**: Environment variable not set correctly

**Solution:**
1. Go to Vercel project → Settings → Environment Variables
2. Verify `VITE_API_BASE_URL` is set to: `https://adverse-weather-safe-segmentation.onrender.com`
3. Make sure it's set for all environments
4. Redeploy (Settings → Deployments → ... → Redeploy)

### Error: "Cannot find module" or "Build errors"
**Cause**: Root directory not set correctly

**Solution:**
1. Project Settings → General
2. Set Root Directory to: `frontend`
3. Save and redeploy

---

## 🔄 Auto-Deploy Setup

Vercel automatically deploys when you push to GitHub!

**How it works:**
1. Push changes to `main` branch
2. Vercel detects the push
3. Automatically builds and deploys
4. You get a new deployment URL

**To disable auto-deploy:**
- Settings → Git → Disable "Automatic deployments"

---

## 📊 Deployment Status

### Backend (Render) ✅
- **URL**: https://adverse-weather-safe-segmentation.onrender.com
- **Status**: Live and tested
- **Health Check**: https://adverse-weather-safe-segmentation.onrender.com/api/ping

### Frontend (Vercel) 🔄
- **Status**: Ready to deploy
- **Configuration**: Complete
- **Environment Variables**: Set during deployment

---

## 🎉 Success Checklist

Once deployed successfully, verify:
- [ ] Frontend URL loads without errors
- [ ] No console errors in browser DevTools
- [ ] Can sign up for new account
- [ ] Can log in with existing account
- [ ] API calls go to Render backend (check Network tab)
- [ ] Can upload RGB and NIR images
- [ ] Predictions work and show results
- [ ] Results display correctly with images

---

## 📝 Final Notes

### Your Deployed URLs:
- **Backend**: https://adverse-weather-safe-segmentation.onrender.com
- **Frontend**: [Your Vercel URL - assigned during deployment]

### Performance:
- **First request**: Might be slow (Render free tier cold start ~30-60s)
- **Subsequent requests**: Much faster
- **Frontend**: Served from CDN (very fast worldwide)

### Free Tier Limits:
- **Render**: 750 hours/month, spins down after 15 min inactivity
- **Vercel**: Unlimited deployments, 100GB bandwidth/month

---

## 🆘 Need Help?

If you encounter issues:
1. Check Vercel build logs for errors
2. Check browser console for runtime errors
3. Verify environment variables are set correctly
4. Test backend separately: https://adverse-weather-safe-segmentation.onrender.com/api/ping
5. Check Network tab in DevTools to see API requests

---

**Good luck with deployment! 🚀**

Once deployed, share your Vercel URL and we can test the full integration!

