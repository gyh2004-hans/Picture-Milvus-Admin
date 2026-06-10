import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8001',
      '/images': 'http://localhost:8001',
      '/pipeline': 'http://localhost:8001',
      '/draw': 'http://localhost:8001',
      '/records': 'http://localhost:8001',
      '/feedback': 'http://localhost:8001',
      '/evaluate': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
      },
    },
  },
});
