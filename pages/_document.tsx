import Document, {
  Html,
  Head,
  Main,
  NextScript,
  type DocumentContext,
  type DocumentInitialProps,
} from "next/document";
import {
  DEFAULT_LOCALE,
  dirOf,
  isAppLocale,
  type AppLocale,
} from "@/i18n/locales";

// Aplica el tema antes del primer paint para evitar el "flash".
// El tema por defecto es claro; solo se activa dark si el usuario lo eligió.
const themeInitScript = `(function(){try{if(localStorage.getItem("theme")==="dark")document.documentElement.classList.add("dark");}catch(e){}})();`;

type Props = DocumentInitialProps & { locale: AppLocale };

export default function MyDocument({ locale }: Props) {
  return (
    <Html lang={locale} dir={dirOf(locale)}>
      <Head />
      <body className="antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

MyDocument.getInitialProps = async (ctx: DocumentContext): Promise<Props> => {
  const initialProps = await Document.getInitialProps(ctx);
  // Con páginas [locale] pre-generadas, el idioma viene del parámetro de ruta
  // (disponible en build para cada página estática). El header x-locale sirve
  // de respaldo para renders dinámicos.
  const fromParam = ctx.query?.locale;
  const raw =
    (Array.isArray(fromParam) ? fromParam[0] : fromParam) ??
    ctx.req?.headers["x-locale"];
  const locale: AppLocale = isAppLocale(raw) ? raw : DEFAULT_LOCALE;
  return { ...initialProps, locale };
};
