import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// `npm run build` emits dist/index.html as a single self-contained file:
// JS and CSS are inlined, so it opens from the filesystem with no server and
// no network at runtime. Fonts are the one exception - see index.css.
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    // Inline every asset regardless of size; nothing may be emitted as a
    // separate file the bundle would then have to fetch.
    assetsInlineLimit: Number.MAX_SAFE_INTEGER,
    cssCodeSplit: false,
  },
})
