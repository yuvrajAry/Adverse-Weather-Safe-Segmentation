# Test Your Live Backend

## Your Backend URL
```
https://adverse-weather-safe-segmentation.onrender.com
```

## Quick Tests

### 1. Health Check (Test in Browser)
Open this URL:
```
https://adverse-weather-safe-segmentation.onrender.com/api/ping
```

**Expected Response:**
```json
{"message": "IDDAW API is running"}
```

### 2. Test Authentication (Using curl or Postman)

**Signup Test:**
```bash
curl -X POST https://adverse-weather-safe-segmentation.onrender.com/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"testuser@example.com","password":"testpass123"}'
```

**Login Test (using pre-created test user):**
```bash
curl -X POST https://adverse-weather-safe-segmentation.onrender.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

---

## ✅ If Tests Pass

Your backend is working! Model predictions might fail until the rebuild completes, but authentication and basic routes work.

---

## 🚀 Next: Deploy Frontend

Once backend tests pass, proceed with frontend deployment to Vercel!
