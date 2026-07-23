// Română (Romanian)
const ro = {
  meta: {
    title:
      "The CV Comedy Podcast - Transformă fiecare CV într-un episod plin de umor cu Gemini",
    description:
      "Încarcă-ți CV-ul și generează un episod plin de umor din The CV Comedy Podcast folosind Google Gemini 3.5 Flash cu TTS multi-speaker",
    ogTitle:
      "The CV Comedy Podcast - Transformă CV-urile în episoade pline de umor",
    ogDescription:
      "Transformă-ți CV-ul într-un episod plin de umor cu Google Gemini 3.5 Flash și TTS Multi-Speaker.",
    keywords:
      "podcast, CV, comedie, Google Gemini, TTS, inteligență artificială, umor",
  },
  header: {
    tagline: "CV-ul tău e invitatul. Gemini scrie roast-ul și îi pune vocile.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Umor inteligent",
    },
  },
  theme: {
    toDark: "Comută la tema întunecată",
    toLight: "Comută la tema luminoasă",
    dark: "Temă întunecată",
    light: "Temă luminoasă",
  },
  language: { label: "Limbă" },
  a11y: { step: "Pasul {number}: {title}" },
  apikey: {
    title: "Cheia ta API",
    inputLabel: "Google AI API Key",
    placeholder: "Lipește aici cheia ta API",
    remember: "Reține în acest browser",
    getKey: "Obține gratuit o cheie API ↗",
    note: "Cheia este folosită direct din browserul tău către API-ul Google. Dacă activezi «Reține», se salvează doar pe acest dispozitiv.",
  },
  cv: {
    title: "CV-ul tău",
    dropDrag: "Trage CV-ul aici sau dă clic pentru a-l selecta",
    dropFormats: "PDF, DOCX, TXT sau imagine (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · dă clic pentru a-l înlocui",
    processing: "Se procesează fișierul...",
    ocring: "Se extrage textul din document cu Gemini (OCR)...",
    ocrButton: "🔍 Extrage textul cu IA (OCR)",
    textLabel: "Textul CV-ului",
    clearFile: "Elimină fișierul",
    placeholderFile:
      "Aici va apărea textul extras din CV-ul tău. Îl poți edita înainte de a genera episodul...",
    placeholderManual: "...sau lipește direct aici textul CV-ului tău",
    errors: {
      reupload: "Încarcă din nou fișierul pentru a extrage textul cu IA.",
      ocrNeedsKey:
        "Introdu cheia ta API (pasul 1) pentru a extrage textul cu OCR (IA).",
      ocrFailed: "Textul nu a putut fi extras cu IA: {reason}",
      ocrEmpty: "Modelul nu a returnat niciun text",
      pdfScanned:
        "Nu s-a putut extrage text din PDF (pare scanat). Introdu cheia ta API și apasă «Extrage textul cu IA (OCR)» sau lipește textul manual.",
      imageNeedsOcr:
        "Pentru a citi CV-ul dintr-o imagine se folosește OCR cu IA. Introdu cheia ta API (pasul 1) și apasă «Extrage textul cu IA (OCR)».",
      docxEmpty: "Nu s-a putut extrage text din DOCX. Lipește CV-ul manual.",
      txtEmpty:
        "Fișierul TXT este gol. Adaugă conținut sau lipește CV-ul manual.",
      unsupported:
        "Tip de fișier neacceptat. Folosește PDF, DOCX, TXT sau o imagine (PNG/JPG/WebP).",
      processFailed: "Eroare la procesarea fișierului: {reason}",
    },
  },
  episode: {
    title: "Episodul tău",
    generate: "🎭 Generează episodul",
    regenerate: "🔁 Generează un episod nou",
    generating: "Se generează episodul...",
    missingKey: "Lipsește cheia ta API (pasul 1).",
    missingCv: "Lipsește textul CV-ului tău (pasul 2).",
    waitVideo: "Așteaptă să se termine exportul videoclipului.",
    scriptWriting: "Se scrie...",
    scriptReady: "Scenariu",
    copy: "📋 Copiază",
    copied: "✓ Copiat",
    share: "📣 Distribuie",
    linkCopied: "✓ Link copiat",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Scenariu (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Anulează videoclipul",
    newEpisode: "✨ Episod nou",
    shareText: "Ascultă episodul plin de umor despre CV-ul meu 🎙️😂",
    progress: {
      video:
        "Se înregistrează videoclipul în timp real (durează cât episodul)...",
      writing: "Se scrie scenariul...",
      writingWith: "Se scrie scenariul cu {model}...",
      recording: "Se înregistrează episodul...",
      recordingPart: "Se înregistrează partea {current} din {total}...",
      partsReady: "{label} ({done}/{total} părți gata)",
      preparingAudio: "Se pregătește audio-ul...",
    },
    errors: {
      scriptFailed: "Eroare la generarea episodului: {reason}",
      scriptEmpty: "Modelul nu a returnat scenariul episodului",
      audioFailed:
        "Eroare la generarea audio-ului: {reason} Ce a fost deja generat se păstrează: poți relua de unde a rămas.",
      noAudio: "Răspunsul nu conține audio",
      videoFailed: "Videoclipul nu a putut fi exportat: {reason}",
      copyFailed: "Nu s-a putut copia în clipboard.",
      resume: "🔁 Reia audio-ul (de la partea {part})",
      retryAudio: "🔁 Reîncearcă audio-ul",
    },
  },
  api: {
    unavailable:
      "Modelele Gemini sunt supraîncărcate în acest moment (eroare 503). De obicei e temporar: așteaptă câteva minute și încearcă din nou.",
    quota:
      "Ai atins limita de cotă a cheii tale API (eroare 429). Așteaptă un minut și încearcă din nou sau verifică-ți planul în Google AI Studio.",
    invalidKey:
      "Cheia API nu este validă sau nu are permisiuni. Verific-o în Google AI Studio.",
    unknown: "Eroare necunoscută",
  },
  player: {
    complete: "Episod complet",
    playBlocked: "Apasă play pentru a continua (partea {current} din {total})",
    waiting: "Se așteaptă următoarea parte a episodului...",
    part: "Partea {current} din {total} · poți asculta în timp ce se generează restul",
    preparing: "Se pregătește prima parte a audio-ului...",
    empty: "Audio-ul episodului va apărea aici",
    cover: "Coperta episodului",
    audioLabel: "Audio-ul episodului",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Te-a făcut să râzi?",
    text: "Susține proiectul pe GitHub Sponsors ca să continue să apară episoade noi.",
    button: "Susține pe GitHub Sponsors",
  },
  prompt: {
    script:
      'Bun venit la The CV Comedy Podcast. Fiecare CV este un nou episod. Ești un duo de comedianți profesioniști specializați în crearea de conținut plin de umor inteligent pentru podcasturi. Creează scenariul pentru un episod de 4-6 minute care critică într-un mod amuzant și sarcastic un CV (CV-ul este invitatul episodului).\n\nFORMAT NECESAR pentru TTS multi-speaker:\nAlex: [textul primei gazde]\nSam: [textul celei de-a doua gazde]\n\nCARACTERISTICILE GAZDELOR:\n- Alex: Analitic și sarcastic, face observații tehnice precise\n- Sam: Spontan și amuzant, face comentarii haioase și observații lejere\n\nELEMENTE DE CRITICAT CU UMOR INTELIGENT:\n- Clișee tipice: "sunt foarte perfecționist", "lucrez bine în echipă"\n- Inconsecvențe temporale sau logice\n- Abilități exagerate: "expert în tot"\n- Descrieri pompoase ale unor joburi banale\n- Obiective profesionale vagi: "caut să cresc profesional"\n- Hobby-uri irelevante sau clișee\n- Greșeli de ortografie sau gramaticale\n\nTON: Sarcastic, dar sofisticat, ca un late-night comedy show. Păstrează umorul inteligent și evită să fii crud.\n\nIMPORTANT:\n- Folosește EXACT "Alex:" și "Sam:" pentru fiecare replică\n- Include pauze naturale cu "[...]"\n- Adaugă accent cu "[énfasis]" unde este potrivit\n- Fă conversația să curgă natural\n- Maximum 3 minute și jumătate. MAXIM\n\nCONTEXT TEMPORAL: Astăzi este {date}. Evaluează datele din CV în raport cu această dată reală: o dată recentă sau ulterioară cunoștințelor tale NU este o inconsecvență temporală.\n\nAnalizează acest CV și creează scenariul episodului în română, și destul de destul de critic (literalmente un roast fără milă):',
    vibes:
      'MATERIAL EXTRA: atașat vine documentul original al CV-ului. Privește-l cu ochi de comediant: poza, designul, tipografia, culorile, "vibe"-ul general. Dacă ceva vizual se pretează la o glumă bună, folosește-l (una sau două mențiuni bine plasate), dar nu-l descrie literal și nu-l transforma în centrul episodului.',
    ttsStyle:
      'Generează audio-ul unui episod de stand-up comedy în română, în format de podcast critic despre CV-uri, cu tonul unui late-night show: sarcastic, dar sofisticat, umor inteligent, evită să fii crud. Folosește exact numele prezentatorilor pentru fiecare replică, include pauze naturale cu "[...]" și accent cu "[énfasis]" unde este potrivit. Fă conversația să curgă natural, ca un show de comedie de seară.',
    ocr: "Extrage și transcrie TOT textul din acest CV (curriculum). Returnează doar textul simplu al documentului, fără comentarii și fără formatare markdown. Păstrează structura (secțiuni, date, liste) cu treceri la rând nou.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplicație dezvoltată cu Vibe Coding (AI) Folosește API-ul Google AI direct din browser.",
    repo: "Depozit pe GitHub",
  },
} as const;

export default ro;
