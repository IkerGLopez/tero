import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import { PrimeVueResolver } from '@primevue/auto-import-resolver'
import Components from 'unplugin-vue-components/vite'
import VueI18nPlugin from '@intlify/unplugin-vue-i18n/vite'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  build: {
    // set to support top level await in src/services/auth.ts
    target: 'es2022',
    sourcemap: true
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: false
      },
      '/manifest.json': {
        target: 'http://localhost:8000',
        changeOrigin: false
      },
      '/mcp': {
        target: 'http://localhost:8000',
        changeOrigin: false,
        ws: true
      },
      '/.well-known': {
        target: 'http://localhost:8000',
        changeOrigin: false
      },
    },
    watch: {
      usePolling: true,
      interval: 100
    }
  },
  plugins: [
    vue(),
    tailwindcss(),
    vueDevTools(),
    Components({
      dts: true,
      dirs: ['src', '../common/src/components'],
      resolvers: [PrimeVueResolver()]
    }),
    VueI18nPlugin()
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
