import { Html, Head, Main, NextScript } from "next/document";

// Aplica el tema antes del primer paint para evitar el "flash".
// El tema por defecto es claro; solo se activa dark si el usuario lo eligió.
const themeInitScript = `(function(){try{if(localStorage.getItem("theme")==="dark")document.documentElement.classList.add("dark");}catch(e){}})();`;

export default function Document() {
  return (
    <Html lang="es">
      <Head />
      <body className="antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
