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
  const header = ctx.req?.headers["x-locale"];
  const locale: AppLocale = isAppLocale(header) ? header : DEFAULT_LOCALE;
  return { ...initialProps, locale };
};
