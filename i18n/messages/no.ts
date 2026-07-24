// Norsk (Norwegian Bokmål)
const no = {
  meta: {
    title:
      "The CV Comedy Podcast - Gjør hver CV om til en humoristisk episode med Gemini",
    description:
      "Last opp CV-en din og lag en humoristisk episode av The CV Comedy Podcast med Google Gemini 3.5 Flash og multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Gjør CV-er om til humoristiske episoder",
    ogDescription:
      "Gjør CV-en din om til en humoristisk episode med Google Gemini 3.5 Flash og Multi-Speaker TTS.",
    keywords:
      "podkast, CV, komedie, Google Gemini, TTS, kunstig intelligens, humor",
  },
  header: {
    tagline: "CV-en din er gjesten. Gemini skriver roasten og gir den stemmer.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Intelligent humor",
    },
  },
  theme: {
    toDark: "Bytt til mørkt tema",
    toLight: "Bytt til lyst tema",
    dark: "Mørkt tema",
    light: "Lyst tema",
  },
  language: { label: "Språk" },
  a11y: {
    step: "Steg {number}: {title}",
    skip: "Hopp til innhold",
  },
  features: {
    toggle: "Hva er dette?",
    heading: "Gjør CV-en din om til en komedieepisode",
    intro:
      "The CV Comedy Podcast tar CV-en din og lager en episode der to programledere kommenterer den med intelligent humor, med manus og stemmer inkludert. Alt behandles i nettleseren din med Google Gemini.",
    items: {
      formats: {
        title: "Alle formater",
        desc: "Last opp CV-en din i PDF, DOCX, TXT eller som bilde; om nødvendig hentes teksten ut med KI-OCR.",
      },
      script: {
        title: "Manus med KI",
        desc: "Gemini skriver et kritisk og morsomt manus, med tonen til et late-night show.",
      },
      voices: {
        title: "Flere stemmer",
        desc: "Episoden fremføres av to programledere med ulike stemmer (multi-speaker TTS).",
      },
      streaming: {
        title: "Lytt mens den lages",
        desc: "Lyden kommer i deler: du kan begynne å lytte uten å vente på at alt blir ferdig.",
      },
      download: {
        title: "Last ned og del",
        desc: "Eksporter episoden som tekst, lyd (.wav) eller video, og del den med ett klikk.",
      },
      privacy: {
        title: "Privat av design",
        desc: "API-nøkkelen din og CV-en din brukes direkte fra nettleseren din, uten å gå via en egen server.",
      },
    },
  },
  apikey: {
    title: "API-nøkkelen din",
    inputLabel: "Google AI API-nøkkel",
    placeholder: "Lim inn API-nøkkelen din her",
    remember: "Husk i denne nettleseren",
    getKey: "Skaff deg en gratis API-nøkkel ↗",
    note: "Nøkkelen brukes direkte fra nettleseren din mot Googles API. Hvis du slår på «Husk», lagres den kun på denne enheten.",
  },
  cv: {
    title: "CV-en din",
    dropDrag: "Dra CV-en din hit eller klikk for å velge",
    dropFormats: "PDF, DOCX, TXT eller bilde (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klikk for å bytte den ut",
    processing: "Behandler fil...",
    ocring: "Henter ut tekst fra dokumentet med Gemini (OCR)...",
    ocrButton: "🔍 Hent ut tekst med KI (OCR)",
    textLabel: "CV-tekst",
    clearFile: "Fjern fil",
    placeholderFile:
      "Her vises teksten som hentes ut fra CV-en din. Du kan redigere den før du lager episoden...",
    placeholderManual: "...eller lim inn teksten fra CV-en din direkte her",
    errors: {
      reupload: "Last opp filen på nytt for å hente ut teksten med KI.",
      ocrNeedsKey:
        "Skriv inn API-nøkkelen din (steg 1) for å hente ut teksten med OCR (KI).",
      ocrFailed: "Klarte ikke å hente ut teksten med KI: {reason}",
      ocrEmpty: "Modellen returnerte ingen tekst",
      pdfScanned:
        "Klarte ikke å hente ut tekst fra PDF-en (den ser skannet ut). Skriv inn API-nøkkelen din og trykk «Hent ut tekst med KI (OCR)», eller lim inn teksten manuelt.",
      imageNeedsOcr:
        "For å lese CV-en fra et bilde brukes OCR med KI. Skriv inn API-nøkkelen din (steg 1) og trykk «Hent ut tekst med KI (OCR)».",
      docxEmpty:
        "Klarte ikke å hente ut tekst fra DOCX-en. Lim inn CV-en manuelt.",
      txtEmpty:
        "TXT-filen er tom. Legg til innhold eller lim inn CV-en manuelt.",
      unsupported:
        "Filtypen støttes ikke. Bruk PDF, DOCX, TXT eller et bilde (PNG/JPG/WebP).",
      processFailed: "Feil ved behandling av fil: {reason}",
    },
  },
  episode: {
    title: "Episoden din",
    generate: "🎭 Lag episode",
    regenerate: "🔁 Lag en ny episode",
    generating: "Lager episode...",
    missingKey: "API-nøkkelen din mangler (steg 1).",
    missingCv: "CV-teksten din mangler (steg 2).",
    waitVideo: "Vent til videoeksporten er ferdig.",
    scriptWriting: "Skriver...",
    scriptReady: "Manus",
    copy: "📋 Kopier",
    copied: "✓ Kopiert",
    share: "📣 Del",
    linkCopied: "✓ Lenke kopiert",
    audioFile: "🎵 Lyd (.wav)",
    scriptFile: "📄 Manus (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Avbryt video",
    newEpisode: "✨ Ny episode",
    shareText: "Hør den humoristiske episoden om CV-en min 🎙️😂",
    progress: {
      video:
        "Tar opp videoen i sanntid (det tar like lang tid som episoden varer)...",
      writing: "Skriver manuset...",
      writingWith: "Skriver manuset med {model}...",
      recording: "Tar opp episoden...",
      recordingPart: "Tar opp del {current} av {total}...",
      partsReady: "{label} ({done}/{total} deler klare)",
      preparingAudio: "Forbereder lyden...",
    },
    errors: {
      scriptFailed: "Feil ved generering av episoden: {reason}",
      scriptEmpty: "Modellen returnerte ikke manuset til episoden",
      audioFailed:
        "Feil ved generering av lyden: {reason} Det som allerede er laget, beholdes: du kan fortsette der du slapp.",
      noAudio: "Svaret inneholder ingen lyd",
      videoFailed: "Klarte ikke å eksportere videoen: {reason}",
      copyFailed: "Klarte ikke å kopiere til utklippstavlen.",
      resume: "🔁 Fortsett lyden (fra del {part})",
      retryAudio: "🔁 Prøv lyden på nytt",
    },
  },
  api: {
    unavailable:
      "Gemini-modellene er overbelastet akkurat nå (feil 503). Det er som regel midlertidig: vent et par minutter og prøv igjen.",
    quota:
      "Du har nådd kvotegrensen for API-nøkkelen din (feil 429). Vent et minutt og prøv igjen, eller sjekk abonnementet ditt i Google AI Studio.",
    invalidKey:
      "API-nøkkelen er ugyldig eller mangler tilgang. Sjekk den i Google AI Studio.",
    unknown: "Ukjent feil",
  },
  player: {
    complete: "Episoden er ferdig",
    playBlocked: "Trykk på spill av for å fortsette (del {current} av {total})",
    waiting: "Venter på neste del av episoden...",
    part: "Del {current} av {total} · du kan lytte mens resten lages",
    preparing: "Forbereder første del av lyden...",
    empty: "Lyden til episoden vises her",
    cover: "Episodeomslag",
    audioLabel: "Episodelyd",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Fikk du deg en latter?",
    text: "Støtt prosjektet på GitHub Sponsors så det fortsetter å komme nye episoder på lufta.",
    button: "Bli sponsor på GitHub Sponsors",
  },
  prompt: {
    script:
      'Velkommen til The CV Comedy Podcast. Hver CV er en ny episode. Du er en duo av profesjonelle komikere som spesialiserer seg på å lage intelligent humoristisk innhold til podkaster. Lag manuset til en episode på 4-6 minutter som på en morsom og sarkastisk måte kritiserer en CV (CV-en er gjesten i episoden).\n\nPÅKREVD FORMAT for multi-speaker TTS:\nAlex: [tekst fra den første verten]\nSam: [tekst fra den andre verten]\n\nVERTENES EGENSKAPER:\n- Alex: Analytisk og sarkastisk, kommer med presise tekniske observasjoner\n- Sam: Spontan og morsom, kommer med morsomme kommentarer og avslappede observasjoner\n\nELEMENTER Å KRITISERE MED INTELLIGENT HUMOR:\n- Typiske klisjeer: "jeg er veldig perfeksjonistisk", "jeg jobber godt i team"\n- Tidsmessige eller logiske inkonsekvenser\n- Overdrevne ferdigheter: "ekspert på alt"\n- Pompøse beskrivelser av enkle jobber\n- Vage karrieremål: "jeg ønsker å vokse profesjonelt"\n- Irrelevante eller klisjéaktige hobbyer\n- Skrive- eller grammatikkfeil\n\nTONE: Sarkastisk, men sofistikert, som et late-night comedy show. Hold humoren intelligent og unngå å være ondskapsfull.\n\nVIKTIG:\n- Bruk NØYAKTIG "Alex:" og "Sam:" for hver replikk\n- Ta med naturlige pauser med "[...]"\n- Legg til ettertrykk med "[énfasis]" der det passer\n- Få samtalen til å flyte naturlig\n- Maks tre og et halvt minutt. MAKS\n\nTIDSKONTEKST: I dag er det {date}. Vurder datoene i CV-en i forhold til denne faktiske datoen: en dato som er nylig eller senere enn det du kjenner til, er IKKE en tidsmessig inkonsekvens.\n\nAnalyser denne CV-en og lag manuset til episoden på norsk, og ganske ganske kritisk (bokstavelig talt en nådeløs roast):',
    vibes:
      'EKSTRA MATERIALE: vedlagt følger originaldokumentet til CV-en. Se på det med komikerblikk: bildet, designet, typografien, fargene, den generelle "vibben". Hvis noe visuelt egner seg for en god vits, bruk det (en eller to velplasserte kommentarer), men ikke beskriv det bokstavelig eller gjør det til sentrum i episoden.',
    ttsStyle:
      'Generer lyden til en standup comedy-episode på norsk, i formatet til en podkast som kritiserer CV-er, med tonen til et late-night show: sarkastisk, men sofistikert, intelligent humor, unngå å være ondskapsfull. Bruk nøyaktig navnene til programlederne for hver replikk, ta med naturlige pauser med "[...]" og ettertrykk med "[énfasis]" der det passer. Få samtalen til å flyte naturlig, som et komedieshow sent på kvelden.',
    ocr: "Hent ut og transkriber ALL teksten i denne CV-en (curriculum vitae). Returner kun ren tekst fra dokumentet, uten kommentarer eller markdown-formatering. Behold strukturen (seksjoner, datoer, lister) med linjeskift.",
  },
  footer: {
    disclaimer:
      "⚠️ Applikasjon utviklet med Vibe Coding (AI) Bruker Google AI API direkte fra nettleseren.",
    repo: "Repositorium på GitHub",
  },
} as const;

export default no;
