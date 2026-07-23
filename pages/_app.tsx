import type { AppProps } from "next/app";
import { useEffect, useState } from "react";
import Head from "next/head";
import { Analytics } from "@vercel/analytics/next";
import {
  DEFAULT_LOCALE,
  Translation,
  dirOf,
  type AppLocale,
} from "@/i18n/translation";
import "./globals.css";

export default function MyApp({ Component, pageProps }: AppProps) {
  // Provider controlado (patrón de los docs): el locale vive en estado React,
  // inicializado con el del request (proxy → x-locale → getServerSideProps).
  // El cambio de idioma es en caliente: setLocale re-renderiza con el nuevo
  // árbol, sin recargar la página.
  const [locale, setLocale] = useState<AppLocale>(
    pageProps.locale ?? DEFAULT_LOCALE
  );
  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = dirOf(locale);
  }, [locale]);
  return (
    <Translation
      locale={locale}
      onLocaleChange={setLocale}
      // El árbol SSR corresponde solo al locale del request; tras un cambio
      // en caliente, cada idioma llega por su propio loader.
      messages={locale === pageProps.locale ? pageProps.messages : undefined}
    >
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <Component {...pageProps} />
      <Analytics />
    </Translation>
  );
}
