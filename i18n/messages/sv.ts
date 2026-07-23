// Svenska (Swedish)
const sv = {
  meta: {
    title:
      "The CV Comedy Podcast - Förvandla varje CV till ett komiskt avsnitt med Gemini",
    description:
      "Ladda upp ditt CV och skapa ett komiskt avsnitt av The CV Comedy Podcast med Google Gemini 3.5 Flash och multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Förvandla CV:n till komiska avsnitt",
    ogDescription:
      "Förvandla ditt CV till ett komiskt avsnitt med Google Gemini 3.5 Flash och Multi-Speaker TTS.",
    keywords:
      "podcast, CV, komedi, Google Gemini, TTS, artificiell intelligens, humor",
  },
  header: {
    tagline: "Ditt CV är gästen. Gemini skriver roasten och ger det röst.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Smart humor",
    },
  },
  theme: {
    toDark: "Byt till mörkt tema",
    toLight: "Byt till ljust tema",
    dark: "Mörkt tema",
    light: "Ljust tema",
  },
  language: { label: "Språk" },
  a11y: { step: "Steg {number}: {title}" },
  apikey: {
    title: "Din API-nyckel",
    inputLabel: "Google AI API-nyckel",
    placeholder: "Klistra in din API-nyckel här",
    remember: "Kom ihåg i den här webbläsaren",
    getKey: "Skaffa en gratis API-nyckel ↗",
    note: "Nyckeln används direkt från din webbläsare mot Googles API. Om du aktiverar ”Kom ihåg” sparas den bara på den här enheten.",
  },
  cv: {
    title: "Ditt CV",
    dropDrag: "Dra ditt CV hit eller klicka för att välja",
    dropFormats: "PDF, DOCX, TXT eller bild (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klicka för att byta ut den",
    processing: "Bearbetar fil...",
    ocring: "Extraherar text från dokumentet med Gemini (OCR)...",
    ocrButton: "🔍 Extrahera text med AI (OCR)",
    textLabel: "CV-text",
    clearFile: "Ta bort fil",
    placeholderFile:
      "Här visas texten som extraherats från ditt CV. Du kan redigera den innan du skapar avsnittet...",
    placeholderManual: "...eller klistra in texten från ditt CV direkt här",
    errors: {
      reupload: "Ladda upp filen igen för att extrahera texten med AI.",
      ocrNeedsKey:
        "Ange din API-nyckel (steg 1) för att extrahera texten med OCR (AI).",
      ocrFailed: "Det gick inte att extrahera texten med AI: {reason}",
      ocrEmpty: "Modellen returnerade ingen text",
      pdfScanned:
        "Det gick inte att extrahera text från PDF:en (den verkar inskannad). Ange din API-nyckel och klicka på ”Extrahera text med AI (OCR)”, eller klistra in texten manuellt.",
      imageNeedsOcr:
        "För att läsa CV:t från en bild används OCR med AI. Ange din API-nyckel (steg 1) och klicka på ”Extrahera text med AI (OCR)”.",
      docxEmpty:
        "Det gick inte att extrahera text från DOCX-filen. Klistra in CV:t manuellt.",
      txtEmpty:
        "TXT-filen är tom. Lägg till innehåll eller klistra in CV:t manuellt.",
      unsupported:
        "Filtypen stöds inte. Använd PDF, DOCX, TXT eller en bild (PNG/JPG/WebP).",
      processFailed: "Fel vid bearbetning av filen: {reason}",
    },
  },
  episode: {
    title: "Ditt avsnitt",
    generate: "🎭 Skapa avsnitt",
    regenerate: "🔁 Skapa ett nytt avsnitt",
    generating: "Skapar avsnitt...",
    missingKey: "Din API-nyckel saknas (steg 1).",
    missingCv: "Texten från ditt CV saknas (steg 2).",
    waitVideo: "Vänta tills videoexporten är klar.",
    scriptWriting: "Skriver...",
    scriptReady: "Manus",
    copy: "📋 Kopiera",
    copied: "✓ Kopierat",
    share: "📣 Dela",
    linkCopied: "✓ Länk kopierad",
    audioFile: "🎵 Ljud (.wav)",
    scriptFile: "📄 Manus (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Avbryt video",
    newEpisode: "✨ Nytt avsnitt",
    shareText: "Lyssna på det komiska avsnittet om mitt CV 🎙️😂",
    progress: {
      video: "Spelar in videon i realtid (tar lika lång tid som avsnittet)...",
      writing: "Skriver manuset...",
      writingWith: "Skriver manuset med {model}...",
      recording: "Spelar in avsnittet...",
      recordingPart: "Spelar in del {current} av {total}...",
      partsReady: "{label} ({done}/{total} delar klara)",
      preparingAudio: "Förbereder ljudet...",
    },
    errors: {
      scriptFailed: "Fel när avsnittet skulle skapas: {reason}",
      scriptEmpty: "Modellen returnerade inget manus för avsnittet",
      audioFailed:
        "Fel när ljudet skulle skapas: {reason} Det som redan skapats sparas: du kan återuppta där du slutade.",
      noAudio: "Svaret innehåller inget ljud",
      videoFailed: "Det gick inte att exportera videon: {reason}",
      copyFailed: "Det gick inte att kopiera till urklipp.",
      resume: "🔁 Återuppta ljudet (från del {part})",
      retryAudio: "🔁 Försök igen med ljudet",
    },
  },
  api: {
    unavailable:
      "Geminis modeller är överbelastade just nu (fel 503). Det brukar vara tillfälligt: vänta ett par minuter och försök igen.",
    quota:
      "Du har nått kvotgränsen för din API-nyckel (fel 429). Vänta en minut och försök igen, eller kontrollera din plan i Google AI Studio.",
    invalidKey:
      "API-nyckeln är ogiltig eller saknar behörighet. Kontrollera den i Google AI Studio.",
    unknown: "Okänt fel",
  },
  player: {
    complete: "Avsnittet är klart",
    playBlocked: "Tryck på play för att fortsätta (del {current} av {total})",
    waiting: "Väntar på nästa del av avsnittet...",
    part: "Del {current} av {total} · du kan lyssna medan resten skapas",
    preparing: "Förbereder den första delen av ljudet...",
    empty: "Avsnittets ljud visas här",
    cover: "Avsnittets omslag",
    audioLabel: "Avsnittets ljud",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Fick det dig att skratta?",
    text: "Stöd projektet på GitHub Sponsors så att fler avsnitt kan fortsätta släppas.",
    button: "Stötta på GitHub Sponsors",
  },
  prompt: {
    script:
      'Välkommen till The CV Comedy Podcast. Varje CV är ett nytt avsnitt. Du är en duo av professionella komiker som specialiserat sig på att skapa smart, humoristiskt innehåll för poddar. Skapa manuset till ett 4-6 minuter långt avsnitt som på ett roligt och sarkastiskt sätt sågar ett CV (CV:t är avsnittets gäst).\n\nOBLIGATORISKT FORMAT för multi-speaker TTS:\nAlex: [text för den första värden]\nSam: [text för den andra värden]\n\nVÄRDARNAS EGENSKAPER:\n- Alex: Analytisk och sarkastisk, gör precisa tekniska iakttagelser\n- Sam: Spontan och rolig, kommer med roliga kommentarer och lättsamma observationer\n\nSAKER ATT SÅGA MED SMART HUMOR:\n- Typiska klichéer: "jag är väldigt perfektionistisk", "jag jobbar bra i team"\n- Tidsmässiga eller logiska inkonsekvenser\n- Överdrivna färdigheter: "expert på allt"\n- Pompösa beskrivningar av enkla jobb\n- Vaga karriärmål: "jag vill växa yrkesmässigt"\n- Irrelevanta eller klichéartade hobbyer\n- Stav- eller grammatikfel\n\nTON: Sarkastisk men sofistikerad, som en late-night comedy show. Håll humorn smart och undvik att vara elak.\n\nVIKTIGT:\n- Använd EXAKT "Alex:" och "Sam:" för varje replik\n- Lägg in naturliga pauser med "[...]"\n- Lägg till betoning med "[énfasis]" där det passar\n- Låt samtalet flyta naturligt\n- Max tre och en halv minut. MAX\n\nTIDSKONTEXT: Idag är det {date}. Bedöm datumen i CV:t i förhållande till detta verkliga datum: ett datum som ligger nära i tiden eller efter din kunskapsgräns är INTE en tidsmässig inkonsekvens.\n\nAnalysera det här CV:t och skriv avsnittets manus på svenska, och var rejält rejält kritisk (bokstavligen en roast utan nåd):',
    vibes:
      'EXTRAMATERIAL: det ursprungliga CV-dokumentet bifogas. Titta på det med en komikers blick: fotot, designen, typsnittet, färgerna, den allmänna "vibben". Om något visuellt duger till ett bra skämt, använd det (en eller två väl placerade kommentarer), men beskriv det inte ordagrant och gör det inte till avsnittets huvudsak.',
    ttsStyle:
      'Skapa ljudet till ett standup comedy-avsnitt på svenska, i formatet av en podd som sågar CV:n, med tonen av en late-night show: sarkastisk men sofistikerad, smart humor, undvik att vara elak. Använd exakt programledarnas namn för varje replik, lägg in naturliga pauser med "[...]" och betoning med "[énfasis]" där det passar. Låt samtalet flyta naturligt, som en komedishow på kvällstid.',
    ocr: "Extrahera och transkribera ALL text i det här CV:t (meritförteckning). Returnera endast dokumentets rena text, utan kommentarer eller markdown-formatering. Behåll strukturen (avsnitt, datum, listor) med radbrytningar.",
  },
  footer: {
    disclaimer:
      "⚠️ App utvecklad med Vibe Coding (AI) Använder Google AI:s API direkt från webbläsaren.",
    repo: "Repository på GitHub",
  },
} as const;

export default sv;
