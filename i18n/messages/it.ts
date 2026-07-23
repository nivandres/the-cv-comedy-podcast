// Italiano (Italian)
const it = {
  meta: {
    title:
      "The CV Comedy Podcast - Trasforma ogni CV in un episodio comico con Gemini",
    description:
      "Carica il tuo CV e genera un episodio comico di The CV Comedy Podcast con Google Gemini 3.5 Flash e TTS multi-speaker",
    ogTitle: "The CV Comedy Podcast - Trasforma i CV in episodi comici",
    ogDescription:
      "Trasforma il tuo CV in un episodio comico con Google Gemini 3.5 Flash e TTS multi-speaker.",
    keywords:
      "podcast, CV, commedia, Google Gemini, TTS, intelligenza artificiale, umorismo",
  },
  header: {
    tagline: "Il tuo CV è l'ospite. Gemini scrive il roast e ci mette le voci.",
    badges: {
      tts: "✨ TTS multi-speaker",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Umorismo intelligente",
    },
  },
  theme: {
    toDark: "Passa al tema scuro",
    toLight: "Passa al tema chiaro",
    dark: "Tema scuro",
    light: "Tema chiaro",
  },
  language: { label: "Lingua" },
  a11y: { step: "Passo {number}: {title}" },
  apikey: {
    title: "La tua API Key",
    inputLabel: "Google AI API Key",
    placeholder: "Incolla qui la tua API Key",
    remember: "Ricorda su questo browser",
    getKey: "Ottieni una API Key gratis ↗",
    note: "La key viene usata direttamente dal tuo browser con l'API di Google. Se attivi «Ricorda», viene salvata solo su questo dispositivo.",
  },
  cv: {
    title: "Il tuo CV",
    dropDrag: "Trascina qui il tuo CV o fai clic per selezionarlo",
    dropFormats: "PDF, DOCX, TXT o immagine (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · fai clic per sostituirlo",
    processing: "Elaborazione del file...",
    ocring: "Estrazione del testo dal documento con Gemini (OCR)...",
    ocrButton: "🔍 Estrai testo con IA (OCR)",
    textLabel: "Testo del CV",
    clearFile: "Rimuovi file",
    placeholderFile:
      "Qui apparirà il testo estratto dal tuo CV. Puoi modificarlo prima di generare l'episodio...",
    placeholderManual: "...oppure incolla qui direttamente il testo del tuo CV",
    errors: {
      reupload: "Ricarica il file per estrarre il testo con l'IA.",
      ocrNeedsKey:
        "Inserisci la tua API Key (passo 1) per estrarre il testo con l'OCR (IA).",
      ocrFailed: "Impossibile estrarre il testo con l'IA: {reason}",
      ocrEmpty: "Il modello non ha restituito alcun testo",
      pdfScanned:
        "Impossibile estrarre il testo dal PDF (sembra scansionato). Inserisci la tua API Key e premi «Estrai testo con IA (OCR)», oppure incolla il testo manualmente.",
      imageNeedsOcr:
        "Per leggere il CV da un'immagine si usa l'OCR con IA. Inserisci la tua API Key (passo 1) e premi «Estrai testo con IA (OCR)».",
      docxEmpty:
        "Impossibile estrarre il testo dal DOCX. Incolla il CV manualmente.",
      txtEmpty:
        "Il file TXT è vuoto. Aggiungi del contenuto o incolla il CV manualmente.",
      unsupported:
        "Tipo di file non supportato. Usa PDF, DOCX, TXT o un'immagine (PNG/JPG/WebP).",
      processFailed: "Errore durante l'elaborazione del file: {reason}",
    },
  },
  episode: {
    title: "Il tuo episodio",
    generate: "🎭 Genera episodio",
    regenerate: "🔁 Genera un nuovo episodio",
    generating: "Generazione dell'episodio...",
    missingKey: "Manca la tua API Key (passo 1).",
    missingCv: "Manca il testo del tuo CV (passo 2).",
    waitVideo: "Attendi il completamento dell'esportazione del video.",
    scriptWriting: "Scrittura in corso...",
    scriptReady: "Copione",
    copy: "📋 Copia",
    copied: "✓ Copiato",
    share: "📣 Condividi",
    linkCopied: "✓ Link copiato",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Copione (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Annulla video",
    newEpisode: "✨ Nuovo episodio",
    shareText: "Ascolta l'episodio comico del mio CV 🎙️😂",
    progress: {
      video:
        "Registrazione del video in tempo reale (dura quanto l'episodio)...",
      writing: "Scrittura del copione...",
      writingWith: "Scrittura del copione con {model}...",
      recording: "Registrazione dell'episodio...",
      recordingPart: "Registrazione della parte {current} di {total}...",
      partsReady: "{label} ({done}/{total} parti pronte)",
      preparingAudio: "Preparazione dell'audio...",
    },
    errors: {
      scriptFailed: "Errore durante la generazione dell'episodio: {reason}",
      scriptEmpty: "Il modello non ha restituito il copione dell'episodio",
      audioFailed:
        "Errore durante la generazione dell'audio: {reason} Quanto già generato viene conservato: puoi riprendere da dove eri rimasto.",
      noAudio: "La risposta non contiene audio",
      videoFailed: "Impossibile esportare il video: {reason}",
      copyFailed: "Impossibile copiare negli appunti.",
      resume: "🔁 Riprendi audio (dalla parte {part})",
      retryAudio: "🔁 Riprova audio",
    },
  },
  api: {
    unavailable:
      "I modelli di Gemini sono sovraccarichi in questo momento (errore 503). Di solito è temporaneo: aspetta un paio di minuti e riprova.",
    quota:
      "Hai raggiunto il limite di quota della tua API Key (errore 429). Aspetta un minuto e riprova, oppure controlla il tuo piano su Google AI Studio.",
    invalidKey:
      "La API Key non è valida o non ha i permessi necessari. Controllala su Google AI Studio.",
    unknown: "Errore sconosciuto",
  },
  player: {
    complete: "Episodio completo",
    playBlocked: "Premi play per continuare (parte {current} di {total})",
    waiting: "In attesa della prossima parte dell'episodio...",
    part: "Parte {current} di {total} · puoi ascoltare mentre viene generato il resto",
    preparing: "Preparazione della prima parte dell'audio...",
    empty: "L'audio dell'episodio apparirà qui",
    cover: "Copertina dell'episodio",
    audioLabel: "Audio dell'episodio",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Ti ha strappato una risata?",
    text: "Sostieni il progetto su GitHub Sponsors così continuiamo a mandare in onda nuovi episodi.",
    button: "Sostieni su GitHub Sponsors",
  },
  prompt: {
    script:
      'Benvenuto a The CV Comedy Podcast. Ogni CV è un nuovo episodio. Siete un duo di comici professionisti specializzati nel creare contenuti umoristici intelligenti per i podcast. Crea il copione per un episodio di 4-6 minuti che critichi in modo divertente e sarcastico un CV (il CV è l\'ospite dell\'episodio).\n\nFORMATO RICHIESTO per il TTS multi-speaker:\nAlex: [testo del primo host]\nSam: [testo del secondo host]\n\nCARATTERISTICHE DEGLI HOST:\n- Alex: Analitico e sarcastico, fa osservazioni tecniche precise\n- Sam: Spontaneo e simpatico, fa battute divertenti e osservazioni casuali\n\nELEMENTI DA CRITICARE CON UMORISMO INTELLIGENTE:\n- Cliché tipici: "sono molto perfezionista", "lavoro bene in team"\n- Incoerenze temporali o logiche\n- Competenze esagerate: "esperto in tutto"\n- Descrizioni pompose di lavori banali\n- Obiettivi professionali vaghi: "cerco di crescere professionalmente"\n- Hobby irrilevanti o banali\n- Errori di ortografia o di grammatica\n\nTONO: Sarcastico ma sofisticato, come un late-night comedy show. Mantieni l\'umorismo intelligente ed evita di essere crudele.\n\nIMPORTANTE:\n- Usa ESATTAMENTE "Alex:" e "Sam:" per ogni battuta\n- Includi pause naturali con "[...]"\n- Aggiungi enfasi con "[énfasis]" dove opportuno\n- Fai in modo che la conversazione scorra in modo naturale\n- Massimo 3 minuti e mezzo. MASSIMO\n\nCONTESTO TEMPORALE: Oggi è {date}. Valuta le date del CV in relazione a questa data reale: una data recente o successiva alle tue conoscenze NON è un\'incoerenza temporale.\n\nAnalizza questo CV e crea il copione dell\'episodio in italiano, e parecchio parecchio critico (letteralmente un roast senza pietà):',
    vibes:
      'MATERIALE EXTRA: in allegato trovi il documento originale del CV. Guardalo con occhio da comico: la foto, il design, il carattere tipografico, i colori, il "vibe" generale. Se qualcosa di visivo si presta a una bella battuta, usalo (una o due menzioni ben piazzate), ma non descriverlo alla lettera né renderlo il centro dell\'episodio.',
    ttsStyle:
      'Genera l\'audio di un episodio di standup comedy in italiano, in formato di podcast critico sui CV, con il tono di un late-night show: sarcastico ma sofisticato, umorismo intelligente, evita di essere crudele. Usa esattamente i nomi dei presentatori per ogni battuta, includi pause naturali con "[...]" ed enfasi con "[énfasis]" dove opportuno. Fai in modo che la conversazione scorra in modo naturale, come uno show di comedy notturno.',
    ocr: "Estrai e trascrivi TUTTO il testo di questo CV (curriculum). Restituisci esclusivamente il testo semplice del documento, senza commenti né formattazione markdown. Conserva la struttura (sezioni, date, elenchi) con gli a capo.",
  },
  footer: {
    disclaimer:
      "⚠️ Applicazione sviluppata con Vibe Coding (AI) Usa l'API di Google AI direttamente dal browser.",
    repo: "Repository su GitHub",
  },
} as const;

export default it;
