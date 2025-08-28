/** @type {import('next').NextConfig} */

// Determine if we're building for production GitHub Pages
const isProd = process.env.NODE_ENV === 'production'
const basePath = isProd ? '/image-annotation' : ''

const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  // GitHub Pages deployment configuration
  basePath: basePath,
  assetPrefix: basePath,
  output: 'export',
  // Pass the base path to the client
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath
  },
  // Allow serving static files from public directory
  async rewrites() {
    return [
      {
        source: '/images/:path*',
        destination: '/images/:path*',
      },
      {
        source: '/annotations/:path*',
        destination: '/annotations/:path*',
      },
    ];
  },
}

module.exports = nextConfig