// Shqip (Albanian)
const sq = {
  meta: {
    title:
      "The CV Comedy Podcast - Kthe çdo CV në një episod humoristik me Gemini",
    description:
      "Ngarko CV-në tënde dhe krijo një episod humoristik të The CV Comedy Podcast me Google Gemini 3.5 Flash dhe TTS multi-folës",
    ogTitle: "The CV Comedy Podcast - Kthe CV-të në episode humoristike",
    ogDescription:
      "Kthe CV-në tënde në një episod humoristik me Google Gemini 3.5 Flash dhe Multi-Speaker TTS.",
    keywords:
      "podcast, CV, komedi, Google Gemini, TTS, inteligjencë artificiale, humor",
  },
  header: {
    tagline:
      "CV-ja jote është i ftuari. Gemini shkruan roast-in dhe i vë zërat.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Humor inteligjent",
    },
  },
  theme: {
    toDark: "Kalo në temën e errët",
    toLight: "Kalo në temën e çelët",
    dark: "Tema e errët",
    light: "Tema e çelët",
  },
  language: { label: "Gjuha" },
  a11y: {
    step: "Hapi {number}: {title}",
    skip: "Kalo te përmbajtja",
  },
  features: {
    toggle: "Çfarë është kjo?",
    heading: "Kthe CV-në tënde në një episod komedie",
    intro:
      "The CV Comedy Podcast merr kurrikulumin tënd dhe krijon një episod ku dy prezantues e komentojnë me humor inteligjent, me skenar dhe zëra të përfshirë. Gjithçka përpunohet në shfletuesin tënd me Google Gemini.",
    items: {
      formats: {
        title: "Çfarëdo formati",
        desc: "Ngarko CV-në tënde në PDF, DOCX, TXT ose si imazh; nëse duhet, teksti nxirret me OCR të IA-s.",
      },
      script: {
        title: "Skenar me IA",
        desc: "Gemini shkruan një skenar kritik dhe argëtues, me tonin e një late-night show.",
      },
      voices: {
        title: "Zëra të shumtë",
        desc: "Episodi zërohet nga dy prezantues me zëra të ndryshëm (multi-speaker TTS).",
      },
      streaming: {
        title: "Dëgjo ndërsa krijohet",
        desc: "Audioja vjen me pjesë: fillon të dëgjosh pa pritur që të mbarojë gjithçka.",
      },
      download: {
        title: "Shkarko dhe shpërnda",
        desc: "Eksporto episodin si tekst, audio (.wav) ose video, dhe shpërndaje me një klik.",
      },
      privacy: {
        title: "Privat nga dizajni",
        desc: "API Key-i yt dhe CV-ja jote përdoren direkt nga shfletuesi yt, pa kaluar nëpër një server tonin.",
      },
    },
  },
  apikey: {
    title: "API Key-i yt",
    inputLabel: "Google AI API Key",
    placeholder: "Ngjit këtu API Key-in tënd",
    remember: "Mbaje mend në këtë shfletues",
    getKey: "Merr një API Key falas ↗",
    note: "Key-i përdoret direkt nga shfletuesi yt përballë API-t të Google-it. Nëse aktivizon «Mbaje mend», ruhet vetëm në këtë pajisje.",
  },
  cv: {
    title: "CV-ja jote",
    dropDrag: "Zvarrit CV-në tënde këtu ose kliko për ta zgjedhur",
    dropFormats: "PDF, DOCX, TXT ose imazh (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · kliko për ta zëvendësuar",
    processing: "Duke përpunuar skedarin...",
    ocring: "Duke nxjerrë tekstin nga dokumenti me Gemini (OCR)...",
    ocrButton: "🔍 Nxirr tekstin me IA (OCR)",
    textLabel: "Teksti i CV-së",
    clearFile: "Hiq skedarin",
    placeholderFile:
      "Këtu do të shfaqet teksti i nxjerrë nga CV-ja jote. Mund ta redaktosh para se të krijosh episodin...",
    placeholderManual: "...ose ngjit këtu direkt tekstin e CV-së tënde",
    errors: {
      reupload: "Ngarko përsëri skedarin për të nxjerrë tekstin me IA.",
      ocrNeedsKey:
        "Fut API Key-in tënd (hapi 1) për të nxjerrë tekstin me OCR (IA).",
      ocrFailed: "Teksti nuk u nxor dot me IA: {reason}",
      ocrEmpty: "Modeli nuk ktheu asnjë tekst",
      pdfScanned:
        "Nuk u nxor dot tekst nga PDF-ja (duket e skanuar). Fut API Key-in tënd dhe shtyp «Nxirr tekstin me IA (OCR)», ose ngjit tekstin manualisht.",
      imageNeedsOcr:
        "Për të lexuar CV-në nga një imazh përdoret OCR me IA. Fut API Key-in tënd (hapi 1) dhe shtyp «Nxirr tekstin me IA (OCR)».",
      docxEmpty: "Nuk u nxor dot tekst nga DOCX-i. Ngjit CV-në manualisht.",
      txtEmpty:
        "Skedari TXT është bosh. Shto përmbajtje ose ngjit CV-në manualisht.",
      unsupported:
        "Lloj skedari i pambështetur. Përdor PDF, DOCX, TXT ose një imazh (PNG/JPG/WebP).",
      processFailed: "Gabim gjatë përpunimit të skedarit: {reason}",
    },
  },
  episode: {
    title: "Episodi yt",
    generate: "🎭 Krijo episodin",
    regenerate: "🔁 Krijo një episod të ri",
    generating: "Duke krijuar episodin...",
    missingKey: "Mungon API Key-i yt (hapi 1).",
    missingCv: "Mungon teksti i CV-së tënde (hapi 2).",
    waitVideo: "Prit të përfundojë eksportimi i videos.",
    scriptWriting: "Duke shkruar...",
    scriptReady: "Skenari",
    copy: "📋 Kopjo",
    copied: "✓ U kopjua",
    share: "📣 Shpërnda",
    linkCopied: "✓ Lidhja u kopjua",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Skenari (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Anulo videon",
    newEpisode: "✨ Episod i ri",
    shareText: "Dëgjo episodin humoristik të CV-së sime 🎙️😂",
    progress: {
      video: "Duke regjistruar videon në kohë reale (zgjat sa vetë episodi)...",
      writing: "Duke shkruar skenarin...",
      writingWith: "Duke shkruar skenarin me {model}...",
      recording: "Duke regjistruar episodin...",
      recordingPart: "Duke regjistruar pjesën {current} nga {total}...",
      partsReady: "{label} ({done}/{total} pjesë gati)",
      preparingAudio: "Duke përgatitur audion...",
    },
    errors: {
      scriptFailed: "Gabim gjatë krijimit të episodit: {reason}",
      scriptEmpty: "Modeli nuk ktheu skenarin e episodit",
      audioFailed:
        "Gabim gjatë krijimit të audios: {reason} Ajo që u krijua ruhet: mund të vazhdosh nga aty ku mbeti.",
      noAudio: "Përgjigja nuk përmban audio",
      videoFailed: "Videoja nuk u eksportua dot: {reason}",
      copyFailed: "Nuk u kopjua dot në clipboard.",
      resume: "🔁 Vazhdo audion (nga pjesa {part})",
      retryAudio: "🔁 Riprovo audion",
    },
  },
  api: {
    unavailable:
      "Modelet e Gemini janë të mbingarkuar për momentin (gabim 503). Zakonisht është e përkohshme: prit ca minuta dhe riprovo.",
    quota:
      "Arrite kufirin e kuotës së API Key-it tënd (gabim 429). Prit një minutë dhe riprovo, ose kontrollo planin tënd në Google AI Studio.",
    invalidKey:
      "API Key-i nuk është i vlefshëm ose s'ka leje. Kontrolloje në Google AI Studio.",
    unknown: "Gabim i panjohur",
  },
  player: {
    complete: "Episodi i plotë",
    playBlocked: "Shtyp play për të vazhduar (pjesa {current} nga {total})",
    waiting: "Duke pritur pjesën tjetër të episodit...",
    part: "Pjesa {current} nga {total} · mund të dëgjosh ndërsa krijohet pjesa tjetër",
    preparing: "Duke përgatitur pjesën e parë të audios...",
    empty: "Audioja e episodit do të shfaqet këtu",
    cover: "Kopertina e episodit",
    audioLabel: "Audioja e episodit",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Të bëri për të qeshur?",
    text: "Mbështet projektin në GitHub Sponsors që të vazhdojnë të dalin episode të reja.",
    button: "Bëhu sponsor në GitHub Sponsors",
  },
  prompt: {
    script:
      'Mirë se vjen te The CV Comedy Podcast. Çdo CV është një episod i ri. Je një dyshe komedianësh profesionistë të specializuar në krijimin e përmbajtjes humoristike inteligjente për podkaste. Krijo skenarin për një episod 4-6 minutash që kritikon në mënyrë argëtuese dhe sarkastike një CV (CV-ja është i ftuari i episodit).\n\nFORMATI I KËRKUAR për TTS multi-folës:\nAlex: [teksti i hostit të parë]\nSam: [teksti i hostit të dytë]\n\nKARAKTERISTIKAT E HOSTAVE:\n- Alex: Analitik dhe sarkastik, bën vëzhgime teknike të sakta\n- Sam: Spontan dhe gazmor, bën komente argëtuese dhe vëzhgime të rastësishme\n\nELEMENTE QË DUHEN KRITIKUAR ME HUMOR INTELIGJENT:\n- Klishe tipike: "jam shumë perfeksionist", "punoj mirë në grup"\n- Mospërputhje kohore ose logjike\n- Aftësi të ekzagjeruara: "ekspert në gjithçka"\n- Përshkrime bombastike të punëve bazike\n- Objektiva profesionale të paqarta: "kërkoj të rritem profesionalisht"\n- Hobi të parëndësishëm ose klishe\n- Gabime drejtshkrimore ose gramatikore\n\nTONI: Sarkastik por i sofistikuar, si një late-night comedy show. Mbaje humorin inteligjent dhe shmang mizorinë.\n\nE RËNDËSISHME:\n- Përdor SAKTËSISHT "Alex:" dhe "Sam:" për çdo ndërhyrje\n- Përfshi pauza natyrale me "[...]"\n- Shto theks me "[énfasis]" aty ku është e përshtatshme\n- Bëje bisedën të rrjedhë natyrshëm\n- Maksimumi 3 minuta e gjysmë. MAKSIMUM\n\nKONTEKSTI KOHOR: Sot është {date}. Vlerëso datat e CV-së në raport me këtë datë reale: një datë e afërt ose pas njohurive të tua NUK është një mospërputhje kohore.\n\nAnalizo këtë CV dhe krijo skenarin e episodit në shqip, dhe mjaft mjaft kritik (fjalë për fjalë një roast pa mëshirë):',
    vibes:
      'MATERIAL EKSTRA: bashkangjitur gjendet dokumenti origjinal i CV-së. Shikoje me sy komediani: fotoja, dizajni, tipografia, ngjyrat, "vibe"-i i përgjithshëm. Nëse diçka vizuale të jep material për një shaka të mirë, përdore (një ose dy përmendje të vendosura mirë), por mos e përshkruaj fjalë për fjalë dhe mos e bëj qendrën e episodit.',
    ttsStyle:
      'Krijo audion e një episodi standup komedie në shqip, në formatin e një podkasti kritik për CV, me tonin e një late-night show: sarkastik por i sofistikuar, humor inteligjent, shmang mizorinë. Përdor saktësisht emrat e prezantuesve për çdo ndërhyrje, përfshi pauza natyrale me "[...]" dhe theks me "[énfasis]" aty ku është e përshtatshme. Bëje bisedën të rrjedhë natyrshëm, si një show komedie nate.',
    ocr: "Nxirr dhe transkripto TË GJITHË tekstin e këtij CV-je (kurrikulum). Kthe vetëm tekstin e thjeshtë të dokumentit, pa komente dhe pa formatim markdown. Ruaj strukturën (seksionet, datat, listat) me kalime rreshti.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikacion i zhvilluar me Vibe Coding (AI) Përdor API-n e Google AI direkt nga shfletuesi.",
    repo: "Depoja në GitHub",
  },
} as const;

export default sq;
