# The CV Comedy Podcast

🎙️ **The CV Comedy Podcast** es una aplicación experimental creada 100% con _full vibe coding_ (programación asistida por IA), diseñada para mostrar el potencial de la inteligencia artificial en el desarrollo creativo y rápido de productos.

## Visita la página web

Visita la página web de The CV Comedy Podcast en https://the-cv-comedy-podcast.vercel.app/

## Archivo principal

> **Nota:** Toda la lógica y la interfaz principal de la aplicación se encuentra en **`pages/index.tsx`**. Si quieres entender, modificar o aprender del código, ¡ese es el archivo clave!

## ¿Qué hace esta app?

- **Convierte cualquier CV (PDF o TXT) en un episodio humorístico de podcast**.
- Utiliza **Google Gemini 2.5 Flash** para:
  - Resumir e interpretar el contenido de CVs, incluso en PDF.
  - Generar un libreto humorístico y crítico, como si el CV fuera el invitado de un late-night show.
  - Crear audio con voces múltiples (multi-speaker TTS), usando personalidades distintas para los hosts.

## ¿Por qué es especial?

- **Full vibe coding**: Todo el código fue generado y refactorizado con ayuda de IA, demostrando cómo se puede crear un producto funcional y divertido en tiempo récord.
- **Gemini 2.5**: Aprovecha las capacidades más avanzadas de Google para análisis de texto, interpretación de PDFs y síntesis de voz multi-speaker.
- **Creatividad y humor**: Cada CV se transforma en un episodio único, con crítica sarcástica y humor inteligente.

## ¿Cómo funciona?

1. **Sube tu CV** (PDF o TXT) o pégalo manualmente.
2. Ingresa tu **API Key de Google AI Studio** (puedes obtenerla gratis en https://aistudio.google.com/app/apikey).
3. La app:
   - Resume e interpreta el CV usando Gemini 2.5.
   - Genera el libreto del episodio (script) con dos hosts de personalidades distintas.
   - Crea el audio del episodio con voces separadas para cada host.
4. Descarga el episodio en texto o escucha el audio generado.

## Requisitos

- Navegador moderno (Chrome, Edge, Firefox, etc.)
- API Key de Google AI Studio

## Instalación y uso local

1. Clona el repositorio:
   ```bash
   git clone <repo-url>
   cd cv-podcast
   ```
2. Instala dependencias:
   ```bash
   npm install
   ```
3. Inicia la app:
   ```bash
   npm run dev
   ```
4. Abre [http://localhost:3000](http://localhost:3000) en tu navegador.

## Notas importantes

- **Experimental**: Esta app es una demo creativa, no apta para producción ni para datos sensibles.
- **Privacidad**: Todo el procesamiento ocurre en el navegador, pero tu API Key se usa directamente en el frontend.
- **Limitaciones**: La calidad del audio y los resultados dependen de la API de Google y pueden variar.

## Créditos

- Programación: 100% asistida por IA (full vibe coding)
- IA: [Google Gemini 2.5](https://aistudio.google.com/)
- Inspiración: El poder de la creatividad + IA

---

¡Disfruta creando episodios únicos con tu CV y explora el futuro del desarrollo asistido por IA!

---

> **Este README también fue generado con IA.**
