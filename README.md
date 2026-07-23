# The CV Comedy Podcast

🎙️ **The CV Comedy Podcast** es una aplicación experimental creada 100% con _full vibe coding_ (programación asistida por IA), diseñada para mostrar el potencial de la inteligencia artificial en el desarrollo creativo y rápido de productos.

## Visita la página web

Visita la página web de The CV Comedy Podcast en https://the-cv-comedy-podcast.vercel.app/

## Arquitectura

El flujo es un **wizard de 3 pasos** (API Key → Tu CV → Tu episodio):

- **`pages/index.tsx`** — orquesta todo el flujo (extracción, libreto, TTS, exportación, compartir).
- **`lib/`** — lógica pura sin React, cubierta por tests (Vitest):
  - `gemini.ts`: modelos, reintentos con fallback (429/503) y segmentación del libreto.
  - `audio.ts`: PCM→WAV, visualizador y exportación de video con `MediaRecorder`.
  - `subtitles.ts`: subtítulos aproximados por peso de texto.
- **`components/`** — UI: sistema de diseño (`ui.tsx`) y reproductor único (`EpisodePlayer.tsx`) que reproduce las partes según llegan y conserva la posición al completarse el episodio.
- **`i18n/`** — internacionalización con [intl-t](https://intl-t.dev): `translation.ts` (árbol de mensajes con loaders por idioma), `navigation.ts` + `proxy.ts` (proxy con `strategy: "request"`: el idioma se resuelve por cookie/`Accept-Language` y llega en el header `x-locale`, sin idioma en la URL) y `messages/` (es/en/pt/fr, incluidos los prompts de generación por idioma).

## ¿Qué hace esta app?

- **Convierte cualquier CV (PDF, DOCX, TXT o imagen) en un episodio humorístico de podcast**.
- Utiliza **Google Gemini 3.5 Flash** para:
  - Resumir e interpretar el contenido de CVs, incluso en PDF.
  - Extraer texto de PDFs escaneados y fotos de CVs con OCR (IA).
  - "Ver" el documento original (foto, diseño, tipografía, vibe general) como material extra para el humor.
  - Generar un libreto humorístico y crítico, como si el CV fuera el invitado de un late-night show.
  - Crear audio con voces múltiples (multi-speaker TTS), usando personalidades distintas para los hosts.
- El audio se genera **por segmentos con progreso real**: puedes empezar a escuchar el episodio mientras el resto se sigue produciendo, y si algo falla se **reanuda desde la última parte completada** (con cadena de fallback entre modelos si alguno está saturado).
- **Subtítulos aproximados (CC)** opcionales sobre el visualizador.
- Puedes **descargar el episodio** como libreto (.txt), audio (.wav) o **video** (portada + visualizador, .mp4/.webm), y **compartirlo** con la Web Share API.
- **Tema claro/oscuro** y modo dev (`?dev=1`) con datos de prueba.
- **Multi-idioma (31 idiomas)** con [intl-t](https://intl-t.dev): árabe, búlgaro, checo, alemán, griego, inglés, español, persa, finés, filipino, francés, hebreo, hindi, croata, indonesio, italiano, japonés, coreano, malayo, noruego, polaco, portugués, rumano, ruso, albanés, sueco, tailandés, turco, vietnamita, chino simplificado y tradicional. La raíz `/` sirve el idioma detectado por cookie/`Accept-Language` (misma URL para todos), el SSR sale traducido, el selector cambia de idioma **en caliente sin recargar**, y cada idioma —prompts de generación incluidos— llega en su propio chunk bajo demanda. Árabe, hebreo y persa se sirven en **RTL** (`dir`).
- **SEO i18n**: cada idioma tiene una **URL canónica crawlable** (`/es`, `/en`, `/ja`, …) servida sin redirect; el `<head>` emite `hreflang` para los 31 idiomas + `x-default`, `canonical`, `og:locale`/`og:locale:alternate` y **JSON-LD** (`WebApplication`); hay **`sitemap.xml`** con alternantes por idioma y **`robots.txt`**.
- Si te gusta, puedes apoyar el proyecto vía [GitHub Sponsors](https://github.com/sponsors/nivandres).

## ¿Por qué es especial?

- **Full vibe coding**: Todo el código fue generado y refactorizado con ayuda de IA, demostrando cómo se puede crear un producto funcional y divertido en tiempo récord.
- **Gemini 3.5**: Aprovecha las capacidades más avanzadas de Google para análisis de texto, interpretación de PDFs y síntesis de voz multi-speaker.
- **Creatividad y humor**: Cada CV se transforma en un episodio único, con crítica sarcástica y humor inteligente.

## ¿Cómo funciona?

1. **Sube tu CV** (PDF, DOCX, TXT o una imagen PNG/JPG/WebP) o pégalo manualmente.
2. Ingresa tu **API Key de Google AI Studio** (puedes obtenerla gratis en https://aistudio.google.com/app/apikey).
3. La app:
   - Resume e interpreta el CV usando Gemini 3.5.
   - Genera el libreto del episodio (script) con dos hosts de personalidades distintas.
   - Crea el audio del episodio con voces separadas para cada host.
4. Escucha el episodio mientras se genera y descárgalo como texto, audio (.wav) o video (.mp4/.webm).

## Requisitos

- Navegador moderno (Chrome, Edge, Firefox, etc.)
- API Key de Google AI Studio

## Instalación y uso local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/nivandres/the-cv-comedy-podcast
   cd the-cv-comedy-podcast
   ```
2. Instala dependencias (Node 20.9+):
   ```bash
   npm install
   ```
3. Inicia la app:
   ```bash
   npm run dev
   ```
4. Abre [http://localhost:3000](http://localhost:3000) en tu navegador.

Otros comandos: `npm test` (Vitest), `npm run lint`, `npm run typecheck`, `npm run format`.

## Notas importantes

- **Experimental**: Esta app es una demo creativa, no apta para producción ni para datos sensibles.
- **Privacidad**: Todo el procesamiento ocurre en el navegador, pero tu API Key se usa directamente en el frontend.
- **Limitaciones**: La calidad del audio y los resultados dependen de la API de Google y pueden variar.

## Créditos

- Programación: 100% asistida por IA (full vibe coding)
- IA: [Google Gemini 3.5](https://aistudio.google.com/)
- Inspiración: El poder de la creatividad + IA

---

¡Disfruta creando episodios únicos con tu CV y explora el futuro del desarrollo asistido por IA!

---

> **Este README también fue generado con IA.**
