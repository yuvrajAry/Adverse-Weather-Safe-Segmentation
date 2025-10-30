import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  root: resolve(__dirname, 'client'),
  server: {
    port: 5173,
    fs: {
      allow: [
        resolve(__dirname, 'client'),
        resolve(__dirname, 'shared'),
      ],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'client'),
      '@shared': resolve(__dirname, 'shared'),
    },
  },
})
