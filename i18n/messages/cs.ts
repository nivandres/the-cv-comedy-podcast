// Čeština (Czech)
const cs = {
  meta: {
    title:
      "The CV Comedy Podcast – Proměňte každý životopis v humorný díl s Gemini",
    description:
      "Nahrajte svůj životopis a vytvořte humorný díl The CV Comedy Podcast pomocí Google Gemini 3.5 Flash s multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast – Proměňte životopisy v humorné díly",
    ogDescription:
      "Proměňte svůj životopis v humorný díl pomocí Google Gemini 3.5 Flash a Multi-Speaker TTS.",
    keywords:
      "podcast, životopis, komedie, Google Gemini, TTS, umělá inteligence, humor",
  },
  header: {
    tagline: "Váš životopis je hostem. Gemini napíše roast a namluví ho.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Chytrý humor",
    },
  },
  theme: {
    toDark: "Přepnout na tmavý motiv",
    toLight: "Přepnout na světlý motiv",
    dark: "Tmavý motiv",
    light: "Světlý motiv",
  },
  language: { label: "Jazyk" },
  a11y: { step: "Krok {number}: {title}" },
  apikey: {
    title: "Váš API klíč",
    inputLabel: "Google AI API klíč",
    placeholder: "Sem vložte svůj API klíč",
    remember: "Zapamatovat v tomto prohlížeči",
    getKey: "Získejte API klíč zdarma ↗",
    note: "Klíč se používá přímo z vašeho prohlížeče proti API Googlu. Pokud zapnete „Zapamatovat“, uloží se jen na tomto zařízení.",
  },
  cv: {
    title: "Váš životopis",
    dropDrag: "Přetáhněte sem svůj životopis nebo klikněte pro výběr",
    dropFormats: "PDF, DOCX, TXT nebo obrázek (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klikněte pro nahrazení",
    processing: "Zpracovávám soubor...",
    ocring: "Získávám text z dokumentu pomocí Gemini (OCR)...",
    ocrButton: "🔍 Získat text pomocí AI (OCR)",
    textLabel: "Text životopisu",
    clearFile: "Odebrat soubor",
    placeholderFile:
      "Zde se zobrazí text získaný z vašeho životopisu. Před vytvořením dílu ho můžete upravit...",
    placeholderManual: "...nebo sem rovnou vložte text svého životopisu",
    errors: {
      reupload: "Nahrajte soubor znovu, abyste získali text pomocí AI.",
      ocrNeedsKey:
        "Zadejte svůj API klíč (krok 1) pro získání textu pomocí OCR (AI).",
      ocrFailed: "Text se nepodařilo získat pomocí AI: {reason}",
      ocrEmpty: "Model nevrátil žádný text",
      pdfScanned:
        "Z PDF se nepodařilo získat text (vypadá naskenovaně). Zadejte svůj API klíč a stiskněte „Získat text pomocí AI (OCR)“, nebo text vložte ručně.",
      imageNeedsOcr:
        "Pro načtení životopisu z obrázku se používá OCR s AI. Zadejte svůj API klíč (krok 1) a stiskněte „Získat text pomocí AI (OCR)“.",
      docxEmpty: "Z DOCX se nepodařilo získat text. Vložte životopis ručně.",
      txtEmpty:
        "Soubor TXT je prázdný. Přidejte obsah nebo vložte životopis ručně.",
      unsupported:
        "Nepodporovaný typ souboru. Použijte PDF, DOCX, TXT nebo obrázek (PNG/JPG/WebP).",
      processFailed: "Chyba při zpracování souboru: {reason}",
    },
  },
  episode: {
    title: "Váš díl",
    generate: "🎭 Vytvořit díl",
    regenerate: "🔁 Vytvořit nový díl",
    generating: "Vytvářím díl...",
    missingKey: "Chybí váš API klíč (krok 1).",
    missingCv: "Chybí text vašeho životopisu (krok 2).",
    waitVideo: "Počkejte, až se dokončí export videa.",
    scriptWriting: "Píšu...",
    scriptReady: "Scénář",
    copy: "📋 Kopírovat",
    copied: "✓ Zkopírováno",
    share: "📣 Sdílet",
    linkCopied: "✓ Odkaz zkopírován",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Scénář (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Zrušit video",
    newEpisode: "✨ Nový díl",
    shareText: "Poslechni si humorný díl o mém životopisu 🎙️😂",
    progress: {
      video: "Nahrávám video v reálném čase (trvá to jako celý díl)...",
      writing: "Píšu scénář...",
      writingWith: "Píšu scénář pomocí {model}...",
      recording: "Nahrávám díl...",
      recordingPart: "Nahrávám část {current} z {total}...",
      partsReady: "{label} ({done}/{total} částí hotovo)",
      preparingAudio: "Připravuji audio...",
    },
    errors: {
      scriptFailed: "Chyba při vytváření dílu: {reason}",
      scriptEmpty: "Model nevrátil scénář dílu",
      audioFailed:
        "Chyba při generování audia: {reason} Co už je vygenerováno, zůstává zachováno: můžete pokračovat tam, kde jste skončili.",
      noAudio: "Odpověď neobsahuje žádné audio",
      videoFailed: "Video se nepodařilo exportovat: {reason}",
      copyFailed: "Nepodařilo se zkopírovat do schránky.",
      resume: "🔁 Pokračovat v audiu (od části {part})",
      retryAudio: "🔁 Zkusit audio znovu",
    },
  },
  api: {
    unavailable:
      "Modely Gemini jsou momentálně přetížené (chyba 503). Obvykle je to dočasné: počkejte pár minut a zkuste to znovu.",
    quota:
      "Dosáhli jste limitu kvóty svého API klíče (chyba 429). Počkejte minutu a zkuste to znovu, nebo zkontrolujte svůj plán v Google AI Studio.",
    invalidKey:
      "API klíč není platný nebo nemá oprávnění. Zkontrolujte ho v Google AI Studio.",
    unknown: "Neznámá chyba",
  },
  player: {
    complete: "Díl kompletní",
    playBlocked: "Stiskněte přehrát pro pokračování (část {current} z {total})",
    waiting: "Čekám na další část dílu...",
    part: "Část {current} z {total} · můžete poslouchat, zatímco se generuje zbytek",
    preparing: "Připravuji první část audia...",
    empty: "Zde se zobrazí audio dílu",
    cover: "Obal dílu",
    audioLabel: "Audio dílu",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Rozesmálo vás to?",
    text: "Podpořte projekt na GitHub Sponsors, aby mohly vznikat další díly.",
    button: "Podpořit na GitHub Sponsors",
  },
  prompt: {
    script:
      'Vítej v The CV Comedy Podcast. Každý životopis je novým dílem. Jsi dvojice profesionálních komiků specializovaných na tvorbu chytrého humorného obsahu pro podcasty. Vytvoř scénář pro díl dlouhý 4–6 minut, který vtipně a sarkasticky zkritizuje životopis (životopis je hostem dílu).\n\nPOŽADOVANÝ FORMÁT pro multi-speaker TTS:\nAlex: [text prvního moderátora]\nSam: [text druhého moderátora]\n\nCHARAKTERISTIKA MODERÁTORŮ:\n- Alex: Analytický a sarkastický, dělá přesné technické postřehy\n- Sam: Spontánní a vtipný, dělá zábavné komentáře a nenucené postřehy\n\nCO KRITIZOVAT S CHYTRÝM HUMOREM:\n- Typická klišé: "jsem velký perfekcionista", "umím dobře pracovat v týmu"\n- Časové nebo logické nesrovnalosti\n- Přehnané dovednosti: "expert na všechno"\n- Nabubřelé popisy obyčejných prací\n- Vágní profesní cíle: "chci profesně růst"\n- Nepodstatné nebo klišovité koníčky\n- Pravopisné nebo gramatické chyby\n\nTÓN: Sarkastický, ale kultivovaný, jako late-night comedy show. Udržuj humor chytrý a vyhni se krutosti.\n\nDŮLEŽITÉ:\n- Používej PŘESNĚ "Alex:" a "Sam:" pro každou repliku\n- Vkládej přirozené pauzy pomocí "[...]"\n- Přidávej důraz pomocí "[énfasis]" tam, kde je to vhodné\n- Ať konverzace plyne přirozeně\n- Maximálně tři a půl minuty. MAXIMÁLNĚ\n\nČASOVÝ KONTEXT: Dnes je {date}. Posuzuj data v životopisu vzhledem k tomuto skutečnému datu: nedávné datum nebo datum pozdější než tvé znalosti NENÍ časová nesrovnalost.\n\nAnalyzuj tento životopis a vytvoř scénář dílu v češtině, a to hodně hodně kritický (doslova nemilosrdný roast):',
    vibes:
      'EXTRA MATERIÁL: v příloze je originální dokument životopisu. Prohlédni si ho okem komika: fotku, design, typografii, barvy, celkový "vibe". Pokud něco vizuálního nabízí dobrý vtip, využij to (jedna nebo dvě dobře umístěné zmínky), ale nepopisuj to doslovně ani z toho nedělej ústřední téma dílu.',
    ttsStyle:
      'Vygeneruj audio dílu stand-up comedy v češtině, ve formátu podcastu kritizujícího životopisy, v tónu late-night show: sarkastický, ale kultivovaný, chytrý humor, vyhni se krutosti. Používej přesně jména moderátorů pro každou repliku, vkládej přirozené pauzy pomocí "[...]" a důraz pomocí "[énfasis]" tam, kde je to vhodné. Ať konverzace plyne přirozeně, jako večerní komediální show.',
    ocr: "Extrahuj a přepiš VEŠKERÝ text z tohoto životopisu (CV). Vrať pouze prostý text dokumentu, bez komentářů a bez formátování markdown. Zachovej strukturu (sekce, data, seznamy) pomocí zalomení řádků.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikace vytvořená pomocí Vibe Coding (AI). Používá Google AI API přímo z prohlížeče.",
    repo: "Repozitář na GitHubu",
  },
} as const;

export default cs;
