/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  // GitHub Pages deployment configuration
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
  assetPrefix: process.env.NEXT_PUBLIC_BASE_PATH || '',
  output: 'export',
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