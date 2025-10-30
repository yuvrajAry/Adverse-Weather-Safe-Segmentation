# AW-SafeSeg UI

A Vite + React + TypeScript SPA for uploading paired RGB + NIR images, sending them to /predict, and viewing segmentation results with download and history. Client-only with optional mock mode.

## Getting Started

1. npm install
2. npm run dev

The app runs at http://localhost:5173.

## Environment

Copy .env.example to .env and adjust as needed:

VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=true

- Set VITE_USE_MOCKS=false to connect to a real backend.

## Scripts

- dev: start dev server
- build: production build
- preview: preview built app
- typecheck: TypeScript validation
- lint: placeholder until ESLint is configured
