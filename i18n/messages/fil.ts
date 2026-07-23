// Filipino
const fil = {
  meta: {
    title:
      "The CV Comedy Podcast - Gawing nakakatawang episode ang bawat CV gamit ang Gemini",
    description:
      "I-upload ang iyong CV at gumawa ng nakakatawang episode ng The CV Comedy Podcast gamit ang Google Gemini 3.5 Flash na may multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Gawing nakakatawang episode ang mga CV",
    ogDescription:
      "Gawing nakakatawang episode ang iyong CV gamit ang Google Gemini 3.5 Flash at Multi-Speaker TTS.",
    keywords:
      "podcast, CV, komedya, Google Gemini, TTS, artificial intelligence, katatawanan",
  },
  header: {
    tagline:
      "Ang iyong CV ang bisita. Ang Gemini ang sumusulat ng roast at nagbibigay ng mga boses.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Matalinong katatawanan",
    },
  },
  theme: {
    toDark: "Lumipat sa madilim na tema",
    toLight: "Lumipat sa maliwanag na tema",
    dark: "Madilim na tema",
    light: "Maliwanag na tema",
  },
  language: { label: "Wika" },
  a11y: { step: "Hakbang {number}: {title}" },
  apikey: {
    title: "Ang iyong API Key",
    inputLabel: "Google AI API Key",
    placeholder: "I-paste dito ang iyong API Key",
    remember: "Tandaan sa browser na ito",
    getKey: "Kumuha ng libreng API Key ↗",
    note: "Direktang ginagamit ang key mula sa iyong browser papunta sa API ng Google. Kung i-o-on mo ang «Tandaan», sa device na ito lang ito naka-save.",
  },
  cv: {
    title: "Ang iyong CV",
    dropDrag: "I-drag dito ang iyong CV o i-click para pumili",
    dropFormats: "PDF, DOCX, TXT o larawan (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · i-click para palitan",
    processing: "Pinoproseso ang file...",
    ocring: "Kinukuha ang teksto mula sa dokumento gamit ang Gemini (OCR)...",
    ocrButton: "🔍 Kunin ang teksto gamit ang AI (OCR)",
    textLabel: "Teksto ng CV",
    clearFile: "Alisin ang file",
    placeholderFile:
      "Dito lilitaw ang tekstong nakuha mula sa iyong CV. Puwede mo itong i-edit bago gawin ang episode...",
    placeholderManual: "...o direktang i-paste dito ang teksto ng iyong CV",
    errors: {
      reupload: "I-upload muli ang file para makuha ang teksto gamit ang AI.",
      ocrNeedsKey:
        "Ilagay ang iyong API Key (hakbang 1) para makuha ang teksto gamit ang OCR (AI).",
      ocrFailed: "Hindi makuha ang teksto gamit ang AI: {reason}",
      ocrEmpty: "Walang isinauling teksto ang modelo",
      pdfScanned:
        "Hindi makuha ang teksto mula sa PDF (mukhang naka-scan). Ilagay ang iyong API Key at pindutin ang «Kunin ang teksto gamit ang AI (OCR)», o i-paste ang teksto nang manu-mano.",
      imageNeedsOcr:
        "Para basahin ang CV mula sa larawan, gumagamit ng OCR na may AI. Ilagay ang iyong API Key (hakbang 1) at pindutin ang «Kunin ang teksto gamit ang AI (OCR)».",
      docxEmpty:
        "Hindi makuha ang teksto mula sa DOCX. I-paste ang CV nang manu-mano.",
      txtEmpty:
        "Walang laman ang TXT file. Magdagdag ng nilalaman o i-paste ang CV nang manu-mano.",
      unsupported:
        "Hindi suportado ang uri ng file. Gumamit ng PDF, DOCX, TXT o larawan (PNG/JPG/WebP).",
      processFailed: "Error sa pagproseso ng file: {reason}",
    },
  },
  episode: {
    title: "Ang iyong episode",
    generate: "🎭 Gumawa ng episode",
    regenerate: "🔁 Gumawa ng bagong episode",
    generating: "Ginagawa ang episode...",
    missingKey: "Kulang ang iyong API Key (hakbang 1).",
    missingCv: "Kulang ang teksto ng iyong CV (hakbang 2).",
    waitVideo: "Hintayin munang matapos ang pag-export ng video.",
    scriptWriting: "Sinusulat...",
    scriptReady: "Iskrip",
    copy: "📋 Kopyahin",
    copied: "✓ Nakopya",
    share: "📣 Ibahagi",
    linkCopied: "✓ Nakopya ang link",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Iskrip (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Kanselahin ang video",
    newEpisode: "✨ Bagong episode",
    shareText: "Pakinggan ang nakakatawang episode ng aking CV 🎙️😂",
    progress: {
      video:
        "Nire-record ang video nang real-time (aabot ito sa haba ng episode)...",
      writing: "Sinusulat ang iskrip...",
      writingWith: "Sinusulat ang iskrip gamit ang {model}...",
      recording: "Nire-record ang episode...",
      recordingPart: "Nire-record ang bahagi {current} ng {total}...",
      partsReady: "{label} ({done}/{total} bahagi ang handa)",
      preparingAudio: "Inihahanda ang audio...",
    },
    errors: {
      scriptFailed: "Error sa paggawa ng episode: {reason}",
      scriptEmpty: "Walang isinauling iskrip ng episode ang modelo",
      audioFailed:
        "Error sa paggawa ng audio: {reason} Nananatili ang nagawa na: puwede kang magpatuloy mula sa huling natigilan.",
      noAudio: "Walang audio ang sagot",
      videoFailed: "Hindi na-export ang video: {reason}",
      copyFailed: "Hindi makopya sa clipboard.",
      resume: "🔁 Ipagpatuloy ang audio (mula sa bahagi {part})",
      retryAudio: "🔁 Subukang muli ang audio",
    },
  },
  api: {
    unavailable:
      "Puno ang mga modelo ng Gemini sa ngayon (error 503). Kadalasan ay pansamantala lang ito: maghintay ng ilang minuto at subukang muli.",
    quota:
      "Naabot mo na ang limitasyon ng quota ng iyong API Key (error 429). Maghintay ng isang minuto at subukang muli, o tingnan ang iyong plano sa Google AI Studio.",
    invalidKey:
      "Hindi wasto ang API Key o walang pahintulot. Suriin ito sa Google AI Studio.",
    unknown: "Hindi kilalang error",
  },
  player: {
    complete: "Kumpleto ang episode",
    playBlocked:
      "Pindutin ang play para magpatuloy (bahagi {current} ng {total})",
    waiting: "Hinihintay ang susunod na bahagi ng episode...",
    part: "Bahagi {current} ng {total} · puwede mong pakinggan habang ginagawa ang iba pa",
    preparing: "Inihahanda ang unang bahagi ng audio...",
    empty: "Dito lilitaw ang audio ng episode",
    cover: "Pabalat ng episode",
    audioLabel: "Audio ng episode",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Napatawa ka ba nito?",
    text: "Suportahan ang proyekto sa GitHub Sponsors para patuloy na may lumalabas na mga episode.",
    button: "Mag-sponsor sa GitHub Sponsors",
  },
  prompt: {
    script:
      'Maligayang pagdating sa The CV Comedy Podcast. Bawat CV ay isang bagong episode. Isa kang duo ng mga propesyonal na komedyante na dalubhasa sa paggawa ng matalino at nakakatawang nilalaman para sa mga podcast. Gumawa ng iskrip para sa isang 4-6 na minutong episode na nakakatuwa at sarkastikong pumupuna sa isang CV (ang CV ang bisita ng episode).\n\nKINAKAILANGANG FORMAT para sa multi-speaker TTS:\nAlex: [teksto ng unang host]\nSam: [teksto ng pangalawang host]\n\nMGA KATANGIAN NG MGA HOST:\n- Alex: Analitikal at sarkastiko, gumagawa ng tumpak na teknikal na mga obserbasyon\n- Sam: Kusang-loob at nakakatawa, gumagawa ng nakakatuwang komento at padalos-dalos na mga obserbasyon\n\nMGA ELEMENTONG PUPUNAHIN NANG MAY MATALINONG KATATAWANAN:\n- Mga tipikal na cliché: "sobrang perpeksiyonista ako", "magaling akong makipagtulungan sa team"\n- Mga di-pagkakatugma sa panahon o lohika\n- Mga pinalabis na kasanayan: "eksperto sa lahat"\n- Mga pa-engrandeng paglalarawan ng mga payak na trabaho\n- Mga malabong layuning propesyonal: "gusto kong lumago sa aking karera"\n- Mga hobby na walang kaugnayan o cliché\n- Mga mali sa ispeling o gramatika\n\nTONO: Sarkastiko pero sopistikado, tulad ng isang late-night comedy show. Panatilihing matalino ang katatawanan at iwasang maging malupit.\n\nMAHALAGA:\n- Gamitin nang EKSAKTO ang "Alex:" at "Sam:" para sa bawat linya\n- Maglagay ng natural na mga paghinto gamit ang "[...]"\n- Magdagdag ng diin gamit ang "[énfasis]" kung saan angkop\n- Gawing natural ang daloy ng usapan\n- Pinakamahaba ay tatlong minuto at kalahati. PINAKAMAHABA\n\nKONTEKSTONG PANAHON: Ngayon ay {date}. Suriin ang mga petsa sa CV kaugnay ng tunay na petsang ito: ang isang kamakailang petsa o petsang mas bago sa iyong kaalaman ay HINDI isang di-pagkakatugma sa panahon.\n\nSuriin ang CV na ito at gumawa ng iskrip ng episode sa Filipino, at sobra-sobrang kritikal (literal na roast na walang awa):',
    vibes:
      'DAGDAG NA MATERYAL: kalakip ang orihinal na dokumento ng CV. Tingnan ito nang may mata ng komedyante: ang litrato, ang disenyo, ang tipograpiya, ang mga kulay, ang pangkalahatang "vibe". Kung may biswal na puwedeng pagbiruan nang maganda, gamitin ito (isa o dalawang banggit na maayos na nakalagay), pero huwag itong ilarawan nang literal o gawing sentro ng episode.',
    ttsStyle:
      'Gumawa ng audio ng isang standup comedy na episode sa Filipino, sa format ng podcast na pumupuna sa mga CV, na may tono ng isang late-night show: sarkastiko pero sopistikado, matalinong katatawanan, iwasang maging malupit. Gamitin nang eksakto ang mga pangalan ng mga host para sa bawat linya, maglagay ng natural na mga paghinto gamit ang "[...]" at diin gamit ang "[énfasis]" kung saan angkop. Gawing natural ang daloy ng usapan, tulad ng isang gabing palabas ng komedya.',
    ocr: "Kunin at i-transcribe ang LAHAT ng teksto ng CV na ito (résumé). Isauli lamang ang plain text ng dokumento, walang komento o markdown na format. Panatilihin ang estruktura (mga seksyon, petsa, listahan) gamit ang mga line break.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikasyong binuo gamit ang Vibe Coding (AI) Gumagamit ng API ng Google AI direkta mula sa browser.",
    repo: "Repository sa GitHub",
  },
} as const;

export default fil;
