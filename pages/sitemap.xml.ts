// Sitemap XML con alternantes hreflang por idioma. Se genera desde la lista
// única de locales (i18n/locales), así no se desincroniza al añadir idiomas.
// El proxy no lo intercepta (el matcher excluye rutas con punto).
import type { GetServerSideProps } from "next";
import { LOCALES, SITE_URL, localeUrl } from "@/i18n/locales";

function buildSitemap(): string {
  // Cada <url> declara el conjunto completo de alternantes (incluido x-default,
  // la raíz que negocia por cookie), como exige la especificación de Google.
  const alternates = [
    `<xhtml:link rel="alternate" hreflang="x-default" href="${SITE_URL}/"/>`,
    ...LOCALES.map(
      (loc) =>
        `<xhtml:link rel="alternate" hreflang="${loc}" href="${localeUrl(loc)}"/>`
    ),
  ].join("");

  const locs = [`${SITE_URL}/`, ...LOCALES.map((loc) => localeUrl(loc))];
  const urls = locs
    .map((loc) => `<url><loc>${loc}</loc>${alternates}</url>`)
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">${urls}</urlset>`;
}

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  res.setHeader("Content-Type", "application/xml; charset=utf-8");
  res.setHeader("Cache-Control", "public, max-age=86400, s-maxage=86400");
  res.write(buildSitemap());
  res.end();
  return { props: {} };
};

export default function Sitemap() {
  return null;
}
