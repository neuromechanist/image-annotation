/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
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