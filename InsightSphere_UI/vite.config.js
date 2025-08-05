import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API requests to the FastAPI server
      '/session': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
})