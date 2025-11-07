import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tsconfigPaths from 'vite-tsconfig-paths'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [
    react(),
    tsconfigPaths({
      root: __dirname, // Match tsconfig baseUrl
    }),
  ],
  root: __dirname, // Root is frontend directory (matches tsconfig baseUrl)
  build: {
    outDir: path.resolve(__dirname, 'dist/spa'),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'client/index.html'),
    },
  },
})
