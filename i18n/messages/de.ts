// Deutsch
const de = {
  meta: {
    title:
      "The CV Comedy Podcast - Verwandle jeden Lebenslauf mit Gemini in eine Comedy-Episode",
    description:
      "Lade deinen Lebenslauf hoch und erstelle mit Google Gemini 3.5 Flash und Multi-Speaker-TTS eine Comedy-Episode von The CV Comedy Podcast",
    ogTitle: "The CV Comedy Podcast - Verwandle Lebensläufe in Comedy-Episoden",
    ogDescription:
      "Verwandle deinen Lebenslauf mit Google Gemini 3.5 Flash und Multi-Speaker-TTS in eine Comedy-Episode.",
    keywords:
      "Podcast, Lebenslauf, CV, Comedy, Google Gemini, TTS, künstliche Intelligenz, Humor",
  },
  header: {
    tagline:
      "Dein Lebenslauf ist der Gast. Gemini schreibt den Roast und liefert die Stimmen gleich mit.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Intelligenter Humor",
    },
  },
  theme: {
    toDark: "Zum dunklen Design wechseln",
    toLight: "Zum hellen Design wechseln",
    dark: "Dunkles Design",
    light: "Helles Design",
  },
  language: { label: "Sprache" },
  a11y: { step: "Schritt {number}: {title}" },
  apikey: {
    title: "Dein API-Key",
    inputLabel: "Google AI API-Key",
    placeholder: "Füge hier deinen API-Key ein",
    remember: "In diesem Browser merken",
    getKey: "Hol dir einen kostenlosen API-Key ↗",
    note: "Der Key wird direkt aus deinem Browser für Anfragen an die Google-API verwendet. Wenn du „Merken“ aktivierst, wird er nur auf diesem Gerät gespeichert.",
  },
  cv: {
    title: "Dein Lebenslauf",
    dropDrag: "Zieh deinen Lebenslauf hierher oder klicke zum Auswählen",
    dropFormats: "PDF, DOCX, TXT oder Bild (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · zum Ersetzen klicken",
    processing: "Datei wird verarbeitet...",
    ocring: "Text wird mit Gemini aus dem Dokument extrahiert (OCR)...",
    ocrButton: "🔍 Text mit KI extrahieren (OCR)",
    textLabel: "Lebenslauf-Text",
    clearFile: "Datei entfernen",
    placeholderFile:
      "Hier erscheint der aus deinem Lebenslauf extrahierte Text. Du kannst ihn bearbeiten, bevor du die Episode erstellst...",
    placeholderManual:
      "...oder füge den Text deines Lebenslaufs direkt hier ein",
    errors: {
      reupload:
        "Lade die Datei erneut hoch, um den Text mit KI zu extrahieren.",
      ocrNeedsKey:
        "Gib deinen API-Key ein (Schritt 1), um den Text mit OCR (KI) zu extrahieren.",
      ocrFailed: "Der Text konnte nicht mit KI extrahiert werden: {reason}",
      ocrEmpty: "Das Modell hat keinen Text zurückgegeben",
      pdfScanned:
        "Aus dem PDF konnte kein Text extrahiert werden (es scheint eingescannt zu sein). Gib deinen API-Key ein und klicke auf „Text mit KI extrahieren (OCR)“, oder füge den Text manuell ein.",
      imageNeedsOcr:
        "Um den Lebenslauf aus einem Bild zu lesen, wird OCR mit KI verwendet. Gib deinen API-Key ein (Schritt 1) und klicke auf „Text mit KI extrahieren (OCR)“.",
      docxEmpty:
        "Aus dem DOCX konnte kein Text extrahiert werden. Füge den Lebenslauf manuell ein.",
      txtEmpty:
        "Die TXT-Datei ist leer. Ergänze Inhalt oder füge den Lebenslauf manuell ein.",
      unsupported:
        "Dateityp nicht unterstützt. Verwende PDF, DOCX, TXT oder ein Bild (PNG/JPG/WebP).",
      processFailed: "Fehler beim Verarbeiten der Datei: {reason}",
    },
  },
  episode: {
    title: "Deine Episode",
    generate: "🎭 Episode erstellen",
    regenerate: "🔁 Neue Episode erstellen",
    generating: "Episode wird erstellt...",
    missingKey: "Dein API-Key fehlt (Schritt 1).",
    missingCv: "Der Text deines Lebenslaufs fehlt (Schritt 2).",
    waitVideo: "Warte, bis der Video-Export abgeschlossen ist.",
    scriptWriting: "Wird geschrieben...",
    scriptReady: "Skript",
    copy: "📋 Kopieren",
    copied: "✓ Kopiert",
    share: "📣 Teilen",
    linkCopied: "✓ Link kopiert",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Skript (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Video abbrechen",
    newEpisode: "✨ Neue Episode",
    shareText: "Hör dir die Comedy-Episode über meinen Lebenslauf an 🎙️😂",
    progress: {
      video:
        "Das Video wird in Echtzeit aufgenommen (dauert so lange wie die Episode)...",
      writing: "Das Skript wird geschrieben...",
      writingWith: "Das Skript wird mit {model} geschrieben...",
      recording: "Die Episode wird aufgenommen...",
      recordingPart: "Teil {current} von {total} wird aufgenommen...",
      partsReady: "{label} ({done}/{total} Teile fertig)",
      preparingAudio: "Audio wird vorbereitet...",
    },
    errors: {
      scriptFailed: "Fehler beim Erstellen der Episode: {reason}",
      scriptEmpty: "Das Modell hat kein Skript für die Episode zurückgegeben",
      audioFailed:
        "Fehler beim Erstellen des Audios: {reason} Das bereits Erstellte bleibt erhalten: Du kannst dort fortsetzen, wo es aufgehört hat.",
      noAudio: "Die Antwort enthält kein Audio",
      videoFailed: "Das Video konnte nicht exportiert werden: {reason}",
      copyFailed: "Konnte nicht in die Zwischenablage kopiert werden.",
      resume: "🔁 Audio fortsetzen (ab Teil {part})",
      retryAudio: "🔁 Audio erneut versuchen",
    },
  },
  api: {
    unavailable:
      "Die Gemini-Modelle sind gerade überlastet (Fehler 503). Das ist meist vorübergehend: Warte ein paar Minuten und versuch es erneut.",
    quota:
      "Du hast das Kontingentlimit deines API-Keys erreicht (Fehler 429). Warte eine Minute und versuch es erneut, oder überprüfe deinen Tarif im Google AI Studio.",
    invalidKey:
      "Der API-Key ist ungültig oder hat keine Berechtigungen. Überprüfe ihn im Google AI Studio.",
    unknown: "Unbekannter Fehler",
  },
  player: {
    complete: "Episode vollständig",
    playBlocked:
      "Drücke auf Play, um fortzufahren (Teil {current} von {total})",
    waiting: "Warte auf den nächsten Teil der Episode...",
    part: "Teil {current} von {total} · du kannst schon hören, während der Rest erstellt wird",
    preparing: "Der erste Teil des Audios wird vorbereitet...",
    empty: "Das Audio der Episode erscheint hier",
    cover: "Cover der Episode",
    audioLabel: "Audio der Episode",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Musstest du lachen?",
    text: "Unterstütze das Projekt auf GitHub Sponsors, damit weiterhin neue Episoden auf Sendung gehen.",
    button: "Auf GitHub Sponsors unterstützen",
  },
  prompt: {
    script:
      'Willkommen bei The CV Comedy Podcast. Jeder Lebenslauf ist eine neue Episode. Du bist ein Duo aus professionellen Comedians, spezialisiert darauf, intelligente, humorvolle Inhalte für Podcasts zu erstellen. Schreibe das Skript für eine 4-6 Minuten lange Episode, die einen Lebenslauf auf witzige und sarkastische Weise auseinandernimmt (der Lebenslauf ist der Gast der Episode).\n\nERFORDERLICHES FORMAT für Multi-Speaker-TTS:\nAlex: [Text des ersten Hosts]\nSam: [Text des zweiten Hosts]\n\nEIGENSCHAFTEN DER HOSTS:\n- Alex: Analytisch und sarkastisch, macht präzise fachliche Beobachtungen\n- Sam: Spontan und witzig, macht lustige Kommentare und lockere Beobachtungen\n\nELEMENTE, DIE MIT INTELLIGENTEM HUMOR ZU KRITISIEREN SIND:\n- Typische Klischees: "ich bin sehr perfektionistisch", "ich arbeite gut im Team"\n- Zeitliche oder logische Widersprüche\n- Übertriebene Fähigkeiten: "Experte für alles"\n- Aufgeblasene Beschreibungen banaler Jobs\n- Vage berufliche Ziele: "ich möchte mich beruflich weiterentwickeln"\n- Irrelevante oder klischeehafte Hobbys\n- Rechtschreib- oder Grammatikfehler\n\nTON: Sarkastisch, aber gehoben, wie eine Late-Night-Comedy-Show. Halte den Humor intelligent und vermeide es, gemein zu werden.\n\nWICHTIG:\n- Verwende EXAKT "Alex:" und "Sam:" für jede Wortmeldung\n- Baue natürliche Pausen mit "[...]" ein\n- Setze Betonung mit "[énfasis]" ein, wo es passt\n- Lass das Gespräch natürlich fließen\n- Maximal dreieinhalb Minuten. MAXIMAL\n\nZEITLICHER KONTEXT: Heute ist {date}. Bewerte die Datumsangaben im Lebenslauf im Verhältnis zu diesem realen Datum: ein aktuelles oder nach deinem Wissensstand liegendes Datum ist KEIN zeitlicher Widerspruch.\n\nAnalysiere diesen Lebenslauf und schreibe das Skript der Episode auf Deutsch, und sei dabei richtig richtig kritisch (buchstäblich ein gnadenloser Roast):',
    vibes:
      'ZUSATZMATERIAL: Im Anhang findest du das Originaldokument des Lebenslaufs. Betrachte es mit dem Auge eines Comedians: das Foto, das Design, die Typografie, die Farben, den allgemeinen "Vibe". Wenn etwas Visuelles sich für einen guten Witz eignet, nutze es (ein oder zwei gut platzierte Erwähnungen), aber beschreibe es nicht wörtlich und mache es nicht zum Mittelpunkt der Episode.',
    ttsStyle:
      'Erzeuge das Audio einer Stand-up-Comedy-Episode auf Deutsch, im Format eines Podcasts, der Lebensläufe kritisiert, im Ton einer Late-Night-Show: sarkastisch, aber gehoben, intelligenter Humor, vermeide es, gemein zu werden. Verwende für jede Wortmeldung exakt die Namen der Moderatoren, baue natürliche Pausen mit "[...]" und Betonung mit "[énfasis]" ein, wo es passt. Lass das Gespräch natürlich fließen, wie eine abendliche Comedy-Show.',
    ocr: "Extrahiere und transkribiere den GESAMTEN Text aus diesem Lebenslauf. Gib ausschließlich den reinen Text des Dokuments zurück, ohne Kommentare und ohne Markdown-Formatierung. Bewahre die Struktur (Abschnitte, Daten, Listen) mit Zeilenumbrüchen.",
  },
  footer: {
    disclaimer:
      "⚠️ App entwickelt mit Vibe Coding (KI). Nutzt die API von Google AI direkt aus dem Browser.",
    repo: "GitHub-Repository",
  },
} as const;

export default de;
