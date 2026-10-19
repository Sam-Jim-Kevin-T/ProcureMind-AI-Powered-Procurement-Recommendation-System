import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Allow public ngrok frontend URLs plus local development hosts.
    allowedHosts: ['.ngrok-free.app', 'localhost', '127.0.0.1'],
  },
});
