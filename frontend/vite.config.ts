import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Manual alias configuration matching tsconfig.json exactly
// tsconfig has: baseUrl: ".", paths: { "@/*": ["./client/*"] }
export default defineConfig({
  plugins: [react()],
  root: __dirname, // frontend directory (matches tsconfig baseUrl: ".")
  resolve: {
    alias: {
      // Match tsconfig paths exactly: "@/*": ["./client/*"]
      '@': path.resolve(__dirname, 'client'),
      '@shared': path.resolve(__dirname, 'shared'),
    },
  },
  build: {
    outDir: path.resolve(__dirname, 'dist/spa'),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'client/index.html'),
    },
  },
})
