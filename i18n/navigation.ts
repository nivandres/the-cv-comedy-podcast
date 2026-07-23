// Routing i18n con el proxy de intl-t (solo se importa desde proxy.ts).
// strategy "request": el idioma se resuelve por request (cookie/Accept-Language)
// y llega a la app en el header x-locale — sin carpeta [locale].
// pathPrefix "optional": la raíz "/" sirve el idioma detectado por cookie sin
// tocar la URL (experiencia limpia, misma para todos), y además /es /en /pt …
// se sirven SIN redirect como URLs canónicas por idioma — indispensable para
// que hreflang/sitemap apunten a URLs reales e indexables (SEO i18n).
// Importa solo @/i18n/locales (datos puros, Edge-safe): no arrastra React.
import { createNavigation } from "intl-t/navigation";
import { DEFAULT_LOCALE, LOCALES } from "@/i18n/locales";

export const { proxy } = createNavigation({
  allowedLocales: LOCALES,
  defaultLocale: DEFAULT_LOCALE,
  strategy: "request",
  pathPrefix: "optional",
});
