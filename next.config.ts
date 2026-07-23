import type { NextConfig } from "next";

// El routing i18n lo maneja el proxy de intl-t (proxy.ts + i18n/navigation.ts)
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
