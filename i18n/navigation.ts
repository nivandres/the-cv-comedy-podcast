// Routing i18n con el proxy de intl-t (solo se importa desde proxy.ts).
// strategy "param": el idioma es un segmento de ruta (carpeta pages/[locale]),
// lo que permite PRE-GENERAR cada idioma como página estática (ISR).
// pathPrefix "optional": la raíz "/" reescribe internamente al idioma detectado
// por cookie/Accept-Language sirviendo su página estática (URL limpia, misma
// para todos), y /es /en /ja … se sirven SIN redirect como URLs canónicas
// crawlables — indispensable para que hreflang/sitemap apunten a URLs reales e
// indexables (SEO i18n). El idioma resuelto llega en el header x-locale.
// Importa solo @/i18n/locales (datos puros, Edge-safe): no arrastra React.
import { createNavigation } from "intl-t/navigation";
import { DEFAULT_LOCALE, LOCALES } from "@/i18n/locales";

export const { proxy } = createNavigation({
  allowedLocales: LOCALES,
  defaultLocale: DEFAULT_LOCALE,
  strategy: "param",
  pathPrefix: "optional",
});
