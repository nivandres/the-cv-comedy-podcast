// Metadatos de idioma puros: sin React ni intl-t, seguros en el Edge (proxy),
// en el servidor y en el cliente. Fuente ÚNICA de la lista de locales para
// translation.ts, navigation.ts, _document, el SEO y el sitemap — así nada se
// desincroniza. Ordenados alfabéticamente por código.
export const LOCALES = [
  "ar",
  "bg",
  "cs",
  "de",
  "el",
  "en",
  "es",
  "fa",
  "fi",
  "fil",
  "fr",
  "he",
  "hi",
  "hr",
  "id",
  "it",
  "ja",
  "ko",
  "ms",
  "no",
  "pl",
  "pt",
  "ro",
  "ru",
  "sq",
  "sv",
  "th",
  "tr",
  "vi",
  "zh-Hans",
  "zh-Hant",
] as const;

export type AppLocale = (typeof LOCALES)[number];

export const DEFAULT_LOCALE: AppLocale = "es";

// Nombre nativo de cada idioma (para el selector)
export const LOCALE_NAMES: Record<AppLocale, string> = {
  ar: "العربية",
  bg: "Български",
  cs: "Čeština",
  de: "Deutsch",
  el: "Ελληνικά",
  en: "English",
  es: "Español",
  fa: "فارسی",
  fi: "Suomi",
  fil: "Filipino",
  fr: "Français",
  he: "עברית",
  hi: "हिन्दी",
  hr: "Hrvatski",
  id: "Bahasa Indonesia",
  it: "Italiano",
  ja: "日本語",
  ko: "한국어",
  ms: "Bahasa Melayu",
  no: "Norsk",
  pl: "Polski",
  pt: "Português",
  ro: "Română",
  ru: "Русский",
  sq: "Shqip",
  sv: "Svenska",
  th: "ไทย",
  tr: "Türkçe",
  vi: "Tiếng Việt",
  "zh-Hans": "简体中文",
  "zh-Hant": "繁體中文",
};

// Idiomas de escritura derecha-a-izquierda
const RTL_LOCALES: readonly AppLocale[] = ["ar", "fa", "he"];

export function dirOf(locale: AppLocale): "rtl" | "ltr" {
  return RTL_LOCALES.includes(locale) ? "rtl" : "ltr";
}

// Etiqueta Open Graph (idioma_TERRITORIO) por locale
export const OG_LOCALES: Record<AppLocale, string> = {
  ar: "ar_AR",
  bg: "bg_BG",
  cs: "cs_CZ",
  de: "de_DE",
  el: "el_GR",
  en: "en_US",
  es: "es_ES",
  fa: "fa_IR",
  fi: "fi_FI",
  fil: "fil_PH",
  fr: "fr_FR",
  he: "he_IL",
  hi: "hi_IN",
  hr: "hr_HR",
  id: "id_ID",
  it: "it_IT",
  ja: "ja_JP",
  ko: "ko_KR",
  ms: "ms_MY",
  no: "nb_NO",
  pl: "pl_PL",
  pt: "pt_BR",
  ro: "ro_RO",
  ru: "ru_RU",
  sq: "sq_AL",
  sv: "sv_SE",
  th: "th_TH",
  tr: "tr_TR",
  vi: "vi_VN",
  "zh-Hans": "zh_CN",
  "zh-Hant": "zh_TW",
};

export const SITE_URL = "https://the-cv-comedy-podcast.vercel.app";

// URL canónica por idioma (crawlable): / es la raíz que negocia por cookie.
export function localeUrl(locale: AppLocale): string {
  return `${SITE_URL}/${locale}`;
}

export function isAppLocale(value: unknown): value is AppLocale {
  return LOCALES.includes(value as AppLocale);
}
