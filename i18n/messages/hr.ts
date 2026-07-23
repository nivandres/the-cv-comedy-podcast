// Hrvatski (Croatian)
const hr = {
  meta: {
    title:
      "The CV Comedy Podcast - Pretvori svaki životopis u urnebesnu epizodu uz Gemini",
    description:
      "Učitaj svoj životopis i generiraj urnebesnu epizodu emisije The CV Comedy Podcast pomoću Google Gemini 3.5 Flash s višeglasnim TTS-om",
    ogTitle: "The CV Comedy Podcast - Pretvori životopise u urnebesne epizode",
    ogDescription:
      "Pretvori svoj životopis u urnebesnu epizodu uz Google Gemini 3.5 Flash i višeglasni TTS.",
    keywords:
      "podcast, životopis, komedija, Google Gemini, TTS, umjetna inteligencija, humor",
  },
  header: {
    tagline: "Tvoj životopis je gost. Gemini piše roast i posuđuje mu glasove.",
    badges: {
      tts: "✨ Višeglasni TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Pametan humor",
    },
  },
  theme: {
    toDark: "Prebaci na tamnu temu",
    toLight: "Prebaci na svijetlu temu",
    dark: "Tamna tema",
    light: "Svijetla tema",
  },
  language: { label: "Jezik" },
  a11y: { step: "Korak {number}: {title}" },
  apikey: {
    title: "Tvoj API ključ",
    inputLabel: "Google AI API ključ",
    placeholder: "Zalijepi ovdje svoj API ključ",
    remember: "Zapamti u ovom pregledniku",
    getKey: "Nabavi besplatan API ključ ↗",
    note: "Ključ se koristi izravno iz tvojeg preglednika prema Googleovom API-ju. Ako uključiš «Zapamti», sprema se samo na ovom uređaju.",
  },
  cv: {
    title: "Tvoj životopis",
    dropDrag: "Povuci svoj životopis ovdje ili klikni za odabir",
    dropFormats: "PDF, DOCX, TXT ili slika (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klikni za zamjenu",
    processing: "Obrada datoteke...",
    ocring: "Izdvajanje teksta iz dokumenta pomoću Geminija (OCR)...",
    ocrButton: "🔍 Izdvoji tekst pomoću AI-ja (OCR)",
    textLabel: "Tekst životopisa",
    clearFile: "Ukloni datoteku",
    placeholderFile:
      "Ovdje će se pojaviti tekst izdvojen iz tvojeg životopisa. Možeš ga urediti prije generiranja epizode...",
    placeholderManual: "...ili zalijepi ovdje tekst svojeg životopisa izravno",
    errors: {
      reupload: "Ponovno učitaj datoteku kako bi izdvojio tekst pomoću AI-ja.",
      ocrNeedsKey:
        "Unesi svoj API ključ (korak 1) za izdvajanje teksta pomoću OCR-a (AI).",
      ocrFailed: "Nije moguće izdvojiti tekst pomoću AI-ja: {reason}",
      ocrEmpty: "Model nije vratio tekst",
      pdfScanned:
        "Nije moguće izdvojiti tekst iz PDF-a (izgleda skenirano). Unesi svoj API ključ i pritisni «Izdvoji tekst pomoću AI-ja (OCR)» ili zalijepi tekst ručno.",
      imageNeedsOcr:
        "Za čitanje životopisa iz slike koristi se OCR s AI-jem. Unesi svoj API ključ (korak 1) i pritisni «Izdvoji tekst pomoću AI-ja (OCR)».",
      docxEmpty:
        "Nije moguće izdvojiti tekst iz DOCX-a. Zalijepi životopis ručno.",
      txtEmpty:
        "TXT datoteka je prazna. Dodaj sadržaj ili zalijepi životopis ručno.",
      unsupported:
        "Vrsta datoteke nije podržana. Koristi PDF, DOCX, TXT ili sliku (PNG/JPG/WebP).",
      processFailed: "Greška pri obradi datoteke: {reason}",
    },
  },
  episode: {
    title: "Tvoja epizoda",
    generate: "🎭 Generiraj epizodu",
    regenerate: "🔁 Generiraj novu epizodu",
    generating: "Generiranje epizode...",
    missingKey: "Nedostaje tvoj API ključ (korak 1).",
    missingCv: "Nedostaje tekst tvojeg životopisa (korak 2).",
    waitVideo: "Pričekaj da završi izvoz videozapisa.",
    scriptWriting: "Pisanje...",
    scriptReady: "Scenarij",
    copy: "📋 Kopiraj",
    copied: "✓ Kopirano",
    share: "📣 Podijeli",
    linkCopied: "✓ Poveznica kopirana",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Scenarij (.txt)",
    video: "🎬 Videozapis",
    cancelVideo: "✕ Otkaži videozapis",
    newEpisode: "✨ Nova epizoda",
    shareText: "Poslušaj urnebesnu epizodu o mojem životopisu 🎙️😂",
    progress: {
      video:
        "Snimanje videozapisa u stvarnom vremenu (traje koliko i epizoda)...",
      writing: "Pisanje scenarija...",
      writingWith: "Pisanje scenarija pomoću {model}...",
      recording: "Snimanje epizode...",
      recordingPart: "Snimanje dijela {current} od {total}...",
      partsReady: "{label} ({done}/{total} dijelova gotovo)",
      preparingAudio: "Priprema zvuka...",
    },
    errors: {
      scriptFailed: "Greška pri generiranju epizode: {reason}",
      scriptEmpty: "Model nije vratio scenarij epizode",
      audioFailed:
        "Greška pri generiranju zvuka: {reason} Ono što je već generirano ostaje sačuvano: možeš nastaviti odande gdje si stao.",
      noAudio: "Odgovor ne sadrži zvuk",
      videoFailed: "Nije moguće izvesti videozapis: {reason}",
      copyFailed: "Nije moguće kopirati u međuspremnik.",
      resume: "🔁 Nastavi zvuk (od dijela {part})",
      retryAudio: "🔁 Pokušaj ponovno zvuk",
    },
  },
  api: {
    unavailable:
      "Gemini modeli su trenutačno preopterećeni (greška 503). Obično je to privremeno: pričekaj nekoliko minuta i pokušaj ponovno.",
    quota:
      "Dosegnuo si ograničenje kvote svojeg API ključa (greška 429). Pričekaj minutu i pokušaj ponovno ili provjeri svoj plan u Google AI Studiju.",
    invalidKey:
      "API ključ nije valjan ili nema dopuštenja. Provjeri ga u Google AI Studiju.",
    unknown: "Nepoznata greška",
  },
  player: {
    complete: "Epizoda dovršena",
    playBlocked: "Pritisni play za nastavak (dio {current} od {total})",
    waiting: "Čekanje sljedećeg dijela epizode...",
    part: "Dio {current} od {total} · možeš slušati dok se generira ostatak",
    preparing: "Priprema prvog dijela zvuka...",
    empty: "Zvuk epizode pojavit će se ovdje",
    cover: "Naslovnica epizode",
    audioLabel: "Zvuk epizode",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Nasmijalo te?",
    text: "Podrži projekt na GitHub Sponsorsu kako bi nove epizode i dalje izlazile.",
    button: "Postani sponzor na GitHub Sponsorsu",
  },
  prompt: {
    script:
      'Dobrodošli u The CV Comedy Podcast. Svaki životopis je nova epizoda. Vi ste dvojac profesionalnih komičara specijaliziranih za stvaranje pametnog humorističnog sadržaja za podcaste. Napiši scenarij za epizodu od 4-6 minuta koja na zabavan i sarkastičan način kritizira životopis (životopis je gost epizode).\n\nOBAVEZNI FORMAT za višeglasni TTS:\nAlex: [tekst prvog voditelja]\nSam: [tekst drugog voditelja]\n\nZNAČAJKE VODITELJA:\n- Alex: Analitičan i sarkastičan, iznosi precizna tehnička zapažanja\n- Sam: Spontan i duhovit, ubacuje zabavne komentare i ležerna zapažanja\n\nELEMENTI ZA KRITIKU S PAMETNIM HUMOROM:\n- Tipični klišeji: "veliki sam perfekcionist", "odlično radim u timu"\n- Vremenske ili logičke nedosljednosti\n- Pretjerane vještine: "stručnjak za sve"\n- Pompozni opisi običnih poslova\n- Nejasni profesionalni ciljevi: "želim profesionalno rasti"\n- Nevažni hobiji ili klišeji\n- Pravopisne ili gramatičke pogreške\n\nTON: Sarkastičan, ali sofisticiran, poput late-night comedy showa. Zadrži pametan humor i izbjegavaj okrutnost.\n\nVAŽNO:\n- Koristi TOČNO "Alex:" i "Sam:" za svaku repliku\n- Uključi prirodne stanke s "[...]"\n- Dodaj naglasak s "[énfasis]" gdje je prikladno\n- Neka razgovor teče prirodno\n- Najviše tri i pol minute. NAJVIŠE\n\nVREMENSKI KONTEKST: Danas je {date}. Procijeni datume iz životopisa u odnosu na ovaj stvarni datum: nedavan datum ili datum nakon granice tvojeg znanja NIJE vremenska nedosljednost.\n\nAnaliziraj ovaj životopis i napiši scenarij epizode na hrvatskom, i to vrlo vrlo kritičan (doslovno nemilosrdan roast):',
    vibes:
      'DODATNI MATERIJAL: u prilogu je izvorni dokument životopisa. Pogledaj ga komičarskim okom: fotografiju, dizajn, tipografiju, boje, opći "vibe". Ako nešto vizualno vrijedi za dobar vic, iskoristi to (jedno ili dva dobro smještena spominjanja), ali nemoj ga doslovno opisivati niti ga učiniti središtem epizode.',
    ttsStyle:
      'Generiraj zvuk epizode standup komedije na hrvatskom, u formatu podcasta koji kritizira životopise, s tonom late-night showa: sarkastično, ali sofisticirano, pametan humor, izbjegavaj okrutnost. Koristi točno imena voditelja za svaku repliku, uključi prirodne stanke s "[...]" i naglaske s "[énfasis]" gdje je prikladno. Neka razgovor teče prirodno, poput večernjeg comedy showa.',
    ocr: "Izdvoji i prepiši SAV tekst iz ovog životopisa (CV-a). Vrati samo čisti tekst dokumenta, bez komentara i markdown formatiranja. Sačuvaj strukturu (odjeljci, datumi, popisi) s prijelomima redaka.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikacija razvijena uz Vibe Coding (AI) Koristi Google AI API izravno iz preglednika.",
    repo: "Repozitorij na GitHubu",
  },
} as const;

export default hr;
