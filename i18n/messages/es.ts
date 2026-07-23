// Español — locale principal y fuente de verdad de la estructura.
// Los demás locales (en/pt/fr) deben replicar exactamente estas claves.
const es = {
  meta: {
    title:
      "The CV Comedy Podcast - Convierte cada CV en un episodio humorístico con Gemini",
    description:
      "Sube tu CV y genera un episodio humorístico de The CV Comedy Podcast usando Google Gemini 3.5 Flash con multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Convierte CVs en episodios humorísticos",
    ogDescription:
      "Convierte tu CV en un episodio humorístico con Google Gemini 3.5 Flash y Multi-Speaker TTS.",
    keywords:
      "podcast, CV, comedia, Google Gemini, TTS, inteligencia artificial, humor",
  },
  header: {
    tagline:
      "Tu CV es el invitado. Gemini escribe el roast y le pone las voces.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Humor inteligente",
    },
  },
  theme: {
    toDark: "Cambiar a tema oscuro",
    toLight: "Cambiar a tema claro",
    dark: "Tema oscuro",
    light: "Tema claro",
  },
  language: { label: "Idioma" },
  a11y: { step: "Paso {number}: {title}" },
  apikey: {
    title: "Tu API Key",
    inputLabel: "Google AI API Key",
    placeholder: "Pega aquí tu API Key",
    remember: "Recordar en este navegador",
    getKey: "Consigue una API Key gratis ↗",
    note: "La key se usa directamente desde tu navegador contra la API de Google. Si activas «Recordar», se guarda solo en este dispositivo.",
  },
  cv: {
    title: "Tu CV",
    dropDrag: "Arrastra tu CV aquí o haz clic para seleccionar",
    dropFormats: "PDF, DOCX, TXT o imagen (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · haz clic para reemplazarlo",
    processing: "Procesando archivo...",
    ocring: "Extrayendo texto del documento con Gemini (OCR)...",
    ocrButton: "🔍 Extraer texto con IA (OCR)",
    textLabel: "Texto del CV",
    clearFile: "Quitar archivo",
    placeholderFile:
      "Aquí aparecerá el texto extraído de tu CV. Puedes editarlo antes de generar el episodio...",
    placeholderManual: "...o pega aquí el texto de tu CV directamente",
    errors: {
      reupload: "Vuelve a subir el archivo para extraer el texto con IA.",
      ocrNeedsKey:
        "Ingresa tu API Key (paso 1) para extraer el texto con OCR (IA).",
      ocrFailed: "No se pudo extraer el texto con IA: {reason}",
      ocrEmpty: "El modelo no devolvió texto",
      pdfScanned:
        "No se pudo extraer texto del PDF (parece escaneado). Ingresa tu API Key y pulsa «Extraer texto con IA (OCR)», o pega el texto manualmente.",
      imageNeedsOcr:
        "Para leer el CV desde una imagen se usa OCR con IA. Ingresa tu API Key (paso 1) y pulsa «Extraer texto con IA (OCR)».",
      docxEmpty: "No se pudo extraer texto del DOCX. Pega el CV manualmente.",
      txtEmpty:
        "El archivo TXT está vacío. Añade contenido o pega el CV manualmente.",
      unsupported:
        "Tipo de archivo no soportado. Usa PDF, DOCX, TXT o una imagen (PNG/JPG/WebP).",
      processFailed: "Error al procesar archivo: {reason}",
    },
  },
  episode: {
    title: "Tu episodio",
    generate: "🎭 Generar episodio",
    regenerate: "🔁 Generar un episodio nuevo",
    generating: "Generando episodio...",
    missingKey: "Falta tu API Key (paso 1).",
    missingCv: "Falta el texto de tu CV (paso 2).",
    waitVideo: "Espera a que termine la exportación del video.",
    scriptWriting: "Escribiendo...",
    scriptReady: "Libreto",
    copy: "📋 Copiar",
    copied: "✓ Copiado",
    share: "📣 Compartir",
    linkCopied: "✓ Enlace copiado",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Libreto (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Cancelar video",
    newEpisode: "✨ Nuevo episodio",
    shareText: "Escucha el episodio humorístico de mi CV 🎙️😂",
    progress: {
      video:
        "Grabando el video en tiempo real (tarda lo que dura el episodio)...",
      writing: "Escribiendo el libreto...",
      writingWith: "Escribiendo el libreto con {model}...",
      recording: "Grabando el episodio...",
      recordingPart: "Grabando parte {current} de {total}...",
      partsReady: "{label} ({done}/{total} partes listas)",
      preparingAudio: "Preparando el audio...",
    },
    errors: {
      scriptFailed: "Error al generar el episodio: {reason}",
      scriptEmpty: "El modelo no devolvió el libreto del episodio",
      audioFailed:
        "Error al generar el audio: {reason} Lo ya generado se conserva: puedes reanudar desde donde quedó.",
      noAudio: "La respuesta no contiene audio",
      videoFailed: "No se pudo exportar el video: {reason}",
      copyFailed: "No se pudo copiar al portapapeles.",
      resume: "🔁 Reanudar audio (desde la parte {part})",
      retryAudio: "🔁 Reintentar audio",
    },
  },
  api: {
    unavailable:
      "Los modelos de Gemini están saturados en este momento (error 503). Suele ser temporal: espera un par de minutos y reintenta.",
    quota:
      "Alcanzaste el límite de cuota de tu API Key (error 429). Espera un minuto y reintenta, o revisa tu plan en Google AI Studio.",
    invalidKey:
      "La API Key no es válida o no tiene permisos. Revísala en Google AI Studio.",
    unknown: "Error desconocido",
  },
  player: {
    complete: "Episodio completo",
    playBlocked: "Pulsa play para continuar (parte {current} de {total})",
    waiting: "Esperando la siguiente parte del episodio...",
    part: "Parte {current} de {total} · puedes escuchar mientras se genera el resto",
    preparing: "Preparando la primera parte del audio...",
    empty: "El audio del episodio aparecerá aquí",
    cover: "Portada del episodio",
    audioLabel: "Audio del episodio",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ ¿Te sacó una risa?",
    text: "Apoya el proyecto en GitHub Sponsors para que sigan saliendo episodios al aire.",
    button: "Patrocinar en GitHub Sponsors",
  },
  prompt: {
    script:
      'Bienvenido a The CV Comedy Podcast. Cada CV es un nuevo episodio. Eres un dúo de comediantes profesionales especializados en crear contenido humorístico inteligente para podcasts. Crea el libreto para un episodio de 4-6 minutos que critique de manera divertida y sarcástica un CV (el CV es el invitado del episodio).\n\nFORMATO REQUERIDO para multi-speaker TTS:\nAlex: [texto del primer host]\nSam: [texto del segundo host]\n\nCARACTERÍSTICAS DE LOS HOSTS:\n- Alex: Analítico y sarcástico, hace observaciones técnicas precisas\n- Sam: Espontáneo y gracioso, hace comentarios divertidos y observaciones casuales\n\nELEMENTOS A CRITICAR CON HUMOR INTELIGENTE:\n- Clichés típicos: "soy muy perfeccionista", "trabajo bien en equipo"\n- Inconsistencias temporales o lógicas\n- Habilidades exageradas: "experto en todo"\n- Descripciones pomposas de trabajos básicos\n- Objetivos profesionales vagos: "busco crecer profesionalmente"\n- Hobbies irrelevantes o clichés\n- Errores ortográficos o gramaticales\n\nTONO: Sarcástico pero sofisticado, como un late-night comedy show. Mantén el humor inteligente y evita ser cruel.\n\nIMPORTANTE:\n- Usa EXACTAMENTE "Alex:" y "Sam:" para cada intervención\n- Incluye pausas naturales con "[...]"\n- Añade énfasis con "[énfasis]" donde sea apropiado\n- Haz que la conversación fluya naturalmente\n- Máximo 3 minutos y medio. MAXIMO\n\nCONTEXTO TEMPORAL: Hoy es {date}. Evalúa las fechas del CV en relación con esta fecha real: una fecha reciente o posterior a tu conocimiento NO es una inconsistencia temporal.\n\nAnaliza este CV y crea el libreto del episodio en español, y bastante bastante crítico (literalmente un roast sin piedad):',
    vibes:
      'MATERIAL EXTRA: adjunto va el documento original del CV. Míralo con ojo de comediante: la foto, el diseño, la tipografía, los colores, el "vibe" general. Si algo visual da para un buen chiste, úsalo (una o dos menciones bien colocadas), pero no lo describas literalmente ni lo conviertas en el centro del episodio.',
    ttsStyle:
      'Genera el audio de un episodio de standup comedy en español, en formato de podcast crítico de CVs, con el tono de un late-night show: sarcástico pero sofisticado, humor inteligente, evita ser cruel. Usa exactamente los nombres de los presentadores para cada intervención, incluye pausas naturales con "[...]" y énfasis con "[énfasis]" donde sea apropiado. Haz que la conversación fluya naturalmente, como un show de comedia nocturno.',
    ocr: "Extrae y transcribe TODO el texto de este CV (currículum). Devuelve únicamente el texto plano del documento, sin comentarios ni formato markdown. Conserva la estructura (secciones, fechas, listas) con saltos de línea.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplicación desarrollada con Vibe Coding (AI) Usa API de Google AI directamente desde el navegador.",
    repo: "Repositorio en GitHub",
  },
} as const;

export default es;
