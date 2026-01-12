import { defineConfig } from 'astro/config'

export default defineConfig({
  site: 'https://zhihanz.github.io',
  output: 'static',
  trailingSlash: 'ignore',
  markdown: {
    shikiConfig: {
      theme: 'one-dark-pro',
      wrap: true
    }
  }
})
