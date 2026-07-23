// Traducciones con intl-t (superficie React, receta canónica para Pages Router).
// Los 31 idiomas son loaders: cada uno es un chunk aparte y el cliente solo
// descarga el que está en pantalla; el árbol SSR viaja por el prop `messages`
// del provider (ver pages/_app.tsx). El tipo del árbol sale del mainLocale (es).
import { createTranslation } from "intl-t/react";
import { DEFAULT_LOCALE, LOCALES, type AppLocale } from "@/i18n/locales";

// Forma del árbol de mensajes (solo tipo, sin coste en runtime): el locale
// principal es la fuente de la estructura y de la inferencia de variables.
type Messages = (typeof import("@/i18n/messages/es"))["default"];

// Loaders por idioma: cada import literal genera un chunk independiente.
export const localeLoaders = {
  ar: () => import("@/i18n/messages/ar").then((m) => m.default),
  bg: () => import("@/i18n/messages/bg").then((m) => m.default),
  cs: () => import("@/i18n/messages/cs").then((m) => m.default),
  de: () => import("@/i18n/messages/de").then((m) => m.default),
  el: () => import("@/i18n/messages/el").then((m) => m.default),
  en: () => import("@/i18n/messages/en").then((m) => m.default),
  es: () => import("@/i18n/messages/es").then((m) => m.default),
  fa: () => import("@/i18n/messages/fa").then((m) => m.default),
  fi: () => import("@/i18n/messages/fi").then((m) => m.default),
  fil: () => import("@/i18n/messages/fil").then((m) => m.default),
  fr: () => import("@/i18n/messages/fr").then((m) => m.default),
  he: () => import("@/i18n/messages/he").then((m) => m.default),
  hi: () => import("@/i18n/messages/hi").then((m) => m.default),
  hr: () => import("@/i18n/messages/hr").then((m) => m.default),
  id: () => import("@/i18n/messages/id").then((m) => m.default),
  it: () => import("@/i18n/messages/it").then((m) => m.default),
  ja: () => import("@/i18n/messages/ja").then((m) => m.default),
  ko: () => import("@/i18n/messages/ko").then((m) => m.default),
  ms: () => import("@/i18n/messages/ms").then((m) => m.default),
  no: () => import("@/i18n/messages/no").then((m) => m.default),
  pl: () => import("@/i18n/messages/pl").then((m) => m.default),
  pt: () => import("@/i18n/messages/pt").then((m) => m.default),
  ro: () => import("@/i18n/messages/ro").then((m) => m.default),
  ru: () => import("@/i18n/messages/ru").then((m) => m.default),
  sq: () => import("@/i18n/messages/sq").then((m) => m.default),
  sv: () => import("@/i18n/messages/sv").then((m) => m.default),
  th: () => import("@/i18n/messages/th").then((m) => m.default),
  tr: () => import("@/i18n/messages/tr").then((m) => m.default),
  vi: () => import("@/i18n/messages/vi").then((m) => m.default),
  "zh-Hans": () => import("@/i18n/messages/zh-Hans").then((m) => m.default),
  "zh-Hant": () => import("@/i18n/messages/zh-Hant").then((m) => m.default),
} satisfies Record<AppLocale, () => Promise<unknown>>;

// Instancia real con los 31 idiomas (runtime). El spread es una copia: el
// provider reemplaza settings.locales[l] con el árbol de `messages`, y
// getServerSideProps necesita los loaders originales intactos.
const app = createTranslation({
  locales: { ...localeLoaders } as unknown as Record<
    AppLocale,
    () => Promise<Messages>
  >,
  allowedLocales: LOCALES,
  mainLocale: DEFAULT_LOCALE,
});

// El tipo del nodo se deriva de UNA sola locale. intl-t tipa cada nodo con sus
// locales hermanas (t.es, t.fr, … recursivas); con 31 idiomas esa recursión
// colapsa el tipo del nodo a `never`. makeTypedNode NUNCA se ejecuta (solo
// aporta un tipo de nodo limpio y callable con la forma de `es`); el runtime es
// `app` con los 31 idiomas y la paridad la garantiza tests/locales.test.ts.
// eslint-disable-next-line @typescript-eslint/no-unused-vars -- solo aporta tipo
function makeTypedNode() {
  return createTranslation({
    locales: { es: () => Promise.resolve({} as Messages) },
    mainLocale: "es" as const,
  });
}
type TypedApp = ReturnType<typeof makeTypedNode>;

export const Translation = app.Translation;
export const useTranslation =
  app.useTranslation as unknown as TypedApp["useTranslation"];
export const t = app.t as unknown as TypedApp["t"];

// Carga de una locale bajo demanda (para el selector): acepta cualquier
// AppLocale y devuelve la rama ya tipada como nodo.
export const loadLocale = (locale: AppLocale) =>
  (app.t as { load: (l: string) => Promise<unknown> }).load(locale) as Promise<
    TypedApp["t"]
  >;

// Reexport de los metadatos de idioma (fuente única en @/i18n/locales)
export {
  LOCALES,
  DEFAULT_LOCALE,
  LOCALE_NAMES,
  OG_LOCALES,
  SITE_URL,
  dirOf,
  localeUrl,
  isAppLocale,
  type AppLocale,
} from "@/i18n/locales";
