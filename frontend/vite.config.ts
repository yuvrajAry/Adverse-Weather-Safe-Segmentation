import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'
import { existsSync } from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Custom plugin to resolve @ aliases with proper file extensions
function aliasResolver() {
  return {
    name: 'alias-resolver',
    resolveId(id, importer) {
      if (id.startsWith('@/')) {
        const relativePath = id.replace('@/', '')
        const clientPath = path.resolve(__dirname, 'client', relativePath)
        
        // Try different extensions
        const extensions = ['.ts', '.tsx', '.js', '.jsx', '.json']
        for (const ext of extensions) {
          const fullPath = clientPath + ext
          if (existsSync(fullPath)) {
            return fullPath
          }
        }
        
        // If no extension found, try as directory with index
        if (existsSync(clientPath)) {
          for (const ext of extensions) {
            const indexPath = path.join(clientPath, 'index' + ext)
            if (existsSync(indexPath)) {
              return indexPath
            }
          }
        }
      }
      if (id.startsWith('@shared/')) {
        const relativePath = id.replace('@shared/', '')
        const sharedPath = path.resolve(__dirname, 'shared', relativePath)
        
        const extensions = ['.ts', '.tsx', '.js', '.jsx', '.json']
        for (const ext of extensions) {
          const fullPath = sharedPath + ext
          if (existsSync(fullPath)) {
            return fullPath
          }
        }
      }
      return null
    },
  }
}

export default defineConfig({
  plugins: [react(), aliasResolver()],
  root: path.resolve(__dirname, 'client'),
  resolve: {
    extensions: ['.mjs', '.js', '.mts', '.ts', '.jsx', '.tsx', '.json'],
  },
  build: {
    outDir: path.resolve(__dirname, 'dist/spa'),
    emptyOutDir: true,
  },
})
