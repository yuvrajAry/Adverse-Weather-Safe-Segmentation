# Vercel Deployment - Complete Guide

## 📋 Prerequisites
- GitHub repository pushed: ✅
- Backend live on Render: ✅ `https://adverse-weather-safe-segmentation.onrender.com`
- Vercel account (sign up at vercel.com with GitHub)

---

## 🚀 Deployment Steps

### Step 1: Go to Vercel
1. Open https://vercel.com in your browser
2. Click **"Sign up"** or **"Login"**
3. Choose **"Continue with GitHub"**
4. Authorize Vercel to access your repositories

---

### Step 2: Create New Project
1. On your Vercel dashboard, click **"Add New..."** (top right)
2. Select **"Project"**
3. You'll see a list of your GitHub repositories

---

### Step 3: Import Your Repository
1. Find: **"Adverse-Weather-Safe-Segmentation"**
2. Click **"Import"** button next to it
3. You'll be taken to the configuration page

---

### Step 4: Configure Project Settings

#### A. Framework Preset
- Vercel should auto-detect **"Vite"**
- If not, select "Vite" from the dropdown

#### B. Root Directory ⚠️ IMPORTANT
- Click **"Edit"** next to "Root Directory"
- Type: `frontend`
- Click **"Continue"**

#### C. Build Settings (verify these are correct)
```
Build Command: npm run build:client
Output Directory: dist/spa
Install Command: npm install
```

---

### Step 5: Environment Variables ⚠️ CRITICAL

Click **"Environment Variables"** section to expand it.

Add TWO variables:

**Variable 1:**
- Name: `VITE_API_BASE_URL`
- Value: `https://adverse-weather-safe-segmentation.onrender.com`
- Environment: Production (default)
- Click "Add"

**Variable 2:**
- Name: `VITE_USE_MOCKS`
- Value: `false`
- Environment: Production (default)
- Click "Add"

**Screenshot example:**
```
VITE_API_BASE_URL = https://adverse-weather-safe-segmentation.onrender.com
VITE_USE_MOCKS = false
```

---

### Step 6: Deploy!
1. Click the big blue **"Deploy"** button at the bottom
2. Vercel will start building your project
3. Watch the build logs (this is optional but interesting!)

---

## ⏱️ Build Process (2-3 minutes)

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
- **Green checkmark** ✓
- **"Visit"** button
- Your app URL (e.g., `https://adverse-weather-123.vercel.app`)

**Click "Visit"** to open your app!

---

## 🧪 Testing Your Deployed App

### 1. Open Your App URL
The URL will be something like:
- `https://your-project-name.vercel.app` or
- `https://your-project-name-username.vercel.app`

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
1. Click "Sign Up" or "Login"
2. Create a new account
3. You should be able to log in successfully

### 4. Test Image Upload (Optional)
- Upload RGB and NIR images
- Check if segmentation works

---

## 🔗 How Frontend Connects to Backend

### The Magic Happens Here:

1. **Environment Variable**: You set `VITE_API_BASE_URL` in Vercel dashboard
2. **Build Time**: Vercel injects this value during build
3. **Runtime**: Frontend code uses it to make API calls

**In the code** (`frontend/client/services/api.ts`):
```javascript
const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8002";
```

All API calls (login, signup, predict) will go to:
```
https://adverse-weather-safe-segmentation.onrender.com/api/...
```

---

## ❌ Common Errors & Solutions

### Error: "Build failed"
**Cause**: Missing dependencies or build errors
**Solution**: Check build logs in Vercel dashboard

### Error: "CORS error" in browser console
**Cause**: Backend not allowing frontend domain
**Solution**: Already fixed! Backend allows all origins (`*`)

### Error: "API calls to localhost:8002"
**Cause**: Environment variables not set correctly
**Solution**: 
1. Go to Vercel project settings
2. Click "Environment Variables"
3. Verify `VITE_API_BASE_URL` is set correctly
4. Redeploy (Settings → Deployments → ... → Redeploy)

### Error: "Cannot find module"
**Cause**: Root directory not set to `frontend`
**Solution**:
1. Project Settings → General
2. Set Root Directory to `frontend`
3. Redeploy

---

## 🔄 Redeploying After Changes

If you need to redeploy:

**Method 1: Auto-deploy (when you push to GitHub)**
```bash
git add .
git commit -m "your changes"
git push origin main
```
Vercel will automatically detect and redeploy!

**Method 2: Manual redeploy**
1. Go to Vercel dashboard
2. Click your project
3. Go to "Deployments" tab
4. Click "..." on latest deployment
5. Click "Redeploy"

---

## 📊 Deployment Status

### Backend (Render) ✅
- URL: https://adverse-weather-safe-segmentation.onrender.com
- Status: Live
- Test: https://adverse-weather-safe-segmentation.onrender.com/api/ping

### Frontend (Vercel) 🔄
- Status: Ready to deploy
- Configuration: Complete
- Environment Variables: Set

---

## 🆘 Still Having Issues?

### Check These:
- [ ] Root Directory is set to `frontend`
- [ ] Environment variables are spelled correctly
- [ ] Both env vars are set (VITE_API_BASE_URL and VITE_USE_MOCKS)
- [ ] Backend URL doesn't have trailing slash
- [ ] Framework is set to "Vite"

### Get Build Logs:
1. In Vercel dashboard
2. Click your project
3. Click "Deployments"
4. Click on the failed deployment
5. Read the error message

**Share the error message with me if you're stuck!**

---

## 🎉 Success Checklist

Once deployed successfully:
- [ ] Frontend URL loads
- [ ] No console errors
- [ ] Can sign up for account
- [ ] Can log in
- [ ] API calls go to Render backend (check Network tab in DevTools)

---

## 📝 Final Notes

### Your Deployed URLs:
- **Backend**: https://adverse-weather-safe-segmentation.onrender.com
- **Frontend**: [Your Vercel URL - will be assigned during deployment]

### Auto-Deploy:
Both services will automatically redeploy when you push to GitHub main branch!

### Performance:
- First load might be slow (Render free tier cold start)
- Subsequent requests will be faster
- Frontend is on CDN (very fast)

---

**Good luck with deployment! Let me know your Vercel URL when it's live! 🚀**
