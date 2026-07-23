// Polski (Polish)
const pl = {
  meta: {
    title:
      "The CV Comedy Podcast - Zamień każde CV w zabawny odcinek dzięki Gemini",
    description:
      "Wgraj swoje CV i wygeneruj zabawny odcinek The CV Comedy Podcast dzięki Google Gemini 3.5 Flash z multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Zamień CV w zabawne odcinki",
    ogDescription:
      "Zamień swoje CV w zabawny odcinek dzięki Google Gemini 3.5 Flash i Multi-Speaker TTS.",
    keywords:
      "podcast, CV, komedia, Google Gemini, TTS, sztuczna inteligencja, humor",
  },
  header: {
    tagline: "Twoje CV to gość odcinka. Gemini pisze roast i podkłada głosy.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Inteligentny humor",
    },
  },
  theme: {
    toDark: "Przełącz na tryb ciemny",
    toLight: "Przełącz na tryb jasny",
    dark: "Tryb ciemny",
    light: "Tryb jasny",
  },
  language: { label: "Język" },
  a11y: { step: "Krok {number}: {title}" },
  apikey: {
    title: "Twój klucz API",
    inputLabel: "Klucz API Google AI",
    placeholder: "Wklej tutaj swój klucz API",
    remember: "Zapamiętaj w tej przeglądarce",
    getKey: "Zdobądź darmowy klucz API ↗",
    note: "Klucz jest używany bezpośrednio z Twojej przeglądarki wobec API Google. Jeśli włączysz „Zapamiętaj”, zostanie zapisany tylko na tym urządzeniu.",
  },
  cv: {
    title: "Twoje CV",
    dropDrag: "Przeciągnij tutaj swoje CV lub kliknij, aby wybrać",
    dropFormats: "PDF, DOCX, TXT lub obraz (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · kliknij, aby zmienić",
    processing: "Przetwarzanie pliku...",
    ocring: "Wyodrębnianie tekstu z dokumentu za pomocą Gemini (OCR)...",
    ocrButton: "🔍 Wyodrębnij tekst za pomocą AI (OCR)",
    textLabel: "Tekst CV",
    clearFile: "Usuń plik",
    placeholderFile:
      "Tutaj pojawi się tekst wyodrębniony z Twojego CV. Możesz go edytować przed wygenerowaniem odcinka...",
    placeholderManual: "...albo wklej tutaj tekst swojego CV bezpośrednio",
    errors: {
      reupload: "Wgraj plik ponownie, aby wyodrębnić tekst za pomocą AI.",
      ocrNeedsKey:
        "Wprowadź swój klucz API (krok 1), aby wyodrębnić tekst za pomocą OCR (AI).",
      ocrFailed: "Nie udało się wyodrębnić tekstu za pomocą AI: {reason}",
      ocrEmpty: "Model nie zwrócił żadnego tekstu",
      pdfScanned:
        "Nie udało się wyodrębnić tekstu z pliku PDF (wygląda na zeskanowany). Wprowadź swój klucz API i naciśnij „Wyodrębnij tekst za pomocą AI (OCR)”, albo wklej tekst ręcznie.",
      imageNeedsOcr:
        "Aby odczytać CV z obrazu, używamy OCR z AI. Wprowadź swój klucz API (krok 1) i naciśnij „Wyodrębnij tekst za pomocą AI (OCR)”.",
      docxEmpty:
        "Nie udało się wyodrębnić tekstu z pliku DOCX. Wklej CV ręcznie.",
      txtEmpty: "Plik TXT jest pusty. Dodaj treść albo wklej CV ręcznie.",
      unsupported:
        "Nieobsługiwany typ pliku. Użyj PDF, DOCX, TXT lub obrazu (PNG/JPG/WebP).",
      processFailed: "Błąd podczas przetwarzania pliku: {reason}",
    },
  },
  episode: {
    title: "Twój odcinek",
    generate: "🎭 Wygeneruj odcinek",
    regenerate: "🔁 Wygeneruj nowy odcinek",
    generating: "Generowanie odcinka...",
    missingKey: "Brakuje Twojego klucza API (krok 1).",
    missingCv: "Brakuje tekstu Twojego CV (krok 2).",
    waitVideo: "Poczekaj, aż zakończy się eksport wideo.",
    scriptWriting: "Pisanie...",
    scriptReady: "Scenariusz",
    copy: "📋 Kopiuj",
    copied: "✓ Skopiowano",
    share: "📣 Udostępnij",
    linkCopied: "✓ Link skopiowany",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Scenariusz (.txt)",
    video: "🎬 Wideo",
    cancelVideo: "✕ Anuluj wideo",
    newEpisode: "✨ Nowy odcinek",
    shareText: "Posłuchaj zabawnego odcinka o moim CV 🎙️😂",
    progress: {
      video:
        "Nagrywanie wideo w czasie rzeczywistym (trwa tyle, ile odcinek)...",
      writing: "Pisanie scenariusza...",
      writingWith: "Pisanie scenariusza za pomocą {model}...",
      recording: "Nagrywanie odcinka...",
      recordingPart: "Nagrywanie części {current} z {total}...",
      partsReady: "{label} ({done}/{total} części gotowych)",
      preparingAudio: "Przygotowywanie audio...",
    },
    errors: {
      scriptFailed: "Błąd podczas generowania odcinka: {reason}",
      scriptEmpty: "Model nie zwrócił scenariusza odcinka",
      audioFailed:
        "Błąd podczas generowania audio: {reason} To, co już wygenerowano, jest zachowane: możesz wznowić od miejsca, w którym się zatrzymało.",
      noAudio: "Odpowiedź nie zawiera audio",
      videoFailed: "Nie udało się wyeksportować wideo: {reason}",
      copyFailed: "Nie udało się skopiować do schowka.",
      resume: "🔁 Wznów audio (od części {part})",
      retryAudio: "🔁 Ponów audio",
    },
  },
  api: {
    unavailable:
      "Modele Gemini są w tej chwili przeciążone (błąd 503). Zwykle jest to tymczasowe: poczekaj kilka minut i spróbuj ponownie.",
    quota:
      "Osiągnięto limit przydziału Twojego klucza API (błąd 429). Poczekaj minutę i spróbuj ponownie albo sprawdź swój plan w Google AI Studio.",
    invalidKey:
      "Klucz API jest nieprawidłowy lub nie ma uprawnień. Sprawdź go w Google AI Studio.",
    unknown: "Nieznany błąd",
  },
  player: {
    complete: "Odcinek kompletny",
    playBlocked: "Naciśnij play, aby kontynuować (część {current} z {total})",
    waiting: "Oczekiwanie na kolejną część odcinka...",
    part: "Część {current} z {total} · możesz słuchać, gdy generowana jest reszta",
    preparing: "Przygotowywanie pierwszej części audio...",
    empty: "Tutaj pojawi się audio odcinka",
    cover: "Okładka odcinka",
    audioLabel: "Audio odcinka",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Rozśmieszyło Cię?",
    text: "Wesprzyj projekt na GitHub Sponsors, żeby kolejne odcinki wciąż trafiały na antenę.",
    button: "Wesprzyj na GitHub Sponsors",
  },
  prompt: {
    script:
      'Witaj w The CV Comedy Podcast. Każde CV to nowy odcinek. Jesteście duetem profesjonalnych komików specjalizujących się w tworzeniu inteligentnych, humorystycznych treści do podcastów. Stwórz scenariusz odcinka trwającego 4-6 minut, który w zabawny i sarkastyczny sposób krytykuje CV (CV jest gościem odcinka).\n\nWYMAGANY FORMAT dla multi-speaker TTS:\nAlex: [tekst pierwszego prowadzącego]\nSam: [tekst drugiego prowadzącego]\n\nCHARAKTERYSTYKA PROWADZĄCYCH:\n- Alex: Analityczny i sarkastyczny, robi precyzyjne uwagi techniczne\n- Sam: Spontaniczny i zabawny, wtrąca śmieszne komentarze i luźne obserwacje\n\nELEMENTY DO SKRYTYKOWANIA Z INTELIGENTNYM HUMOREM:\n- Typowe frazesy: "jestem perfekcjonistą", "dobrze pracuję w zespole"\n- Nieścisłości czasowe lub logiczne\n- Przesadzone umiejętności: "ekspert od wszystkiego"\n- Napuszone opisy prostych stanowisk\n- Mętne cele zawodowe: "chcę się rozwijać zawodowo"\n- Nieistotne lub oklepane hobby\n- Błędy ortograficzne lub gramatyczne\n\nTON: Sarkastyczny, ale wyrafinowany, jak late-night comedy show. Zachowaj inteligentny humor i unikaj okrucieństwa.\n\nWAŻNE:\n- Używaj DOKŁADNIE "Alex:" i "Sam:" przy każdej wypowiedzi\n- Wstawiaj naturalne pauzy za pomocą "[...]"\n- Dodawaj nacisk za pomocą "[énfasis]" tam, gdzie to pasuje\n- Spraw, by rozmowa płynęła naturalnie\n- Maksymalnie trzy i pół minuty. MAKSYMALNIE\n\nKONTEKST CZASOWY: Dziś jest {date}. Oceniaj daty w CV w odniesieniu do tej rzeczywistej daty: data niedawna lub późniejsza niż Twoja wiedza NIE jest nieścisłością czasową.\n\nPrzeanalizuj to CV i stwórz scenariusz odcinka po polsku, i bądź bardzo, bardzo krytyczny (dosłownie roast bez litości):',
    vibes:
      'MATERIAŁ DODATKOWY: w załączniku znajduje się oryginalny dokument CV. Spójrz na niego okiem komika: zdjęcie, układ, typografia, kolory, ogólny "vibe". Jeśli coś wizualnego nadaje się na dobry żart, wykorzystaj to (jedna lub dwie dobrze wstawione wzmianki), ale nie opisuj tego dosłownie ani nie rób z tego głównego wątku odcinka.',
    ttsStyle:
      'Wygeneruj audio odcinka stand-up comedy po polsku, w formacie podcastu krytykującego CV, w tonie late-night show: sarkastycznie, ale wyrafinowanie, z inteligentnym humorem, unikając okrucieństwa. Używaj dokładnie imion prowadzących przy każdej wypowiedzi, wstawiaj naturalne pauzy za pomocą "[...]" i nacisk za pomocą "[énfasis]" tam, gdzie to pasuje. Spraw, by rozmowa płynęła naturalnie, jak wieczorny show komediowy.',
    ocr: "Wyodrębnij i przepisz CAŁY tekst z tego CV (życiorysu). Zwróć wyłącznie zwykły tekst dokumentu, bez komentarzy i bez formatowania markdown. Zachowaj strukturę (sekcje, daty, listy) za pomocą podziałów wierszy.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikacja stworzona z użyciem Vibe Coding (AI). Korzysta z API Google AI bezpośrednio z przeglądarki.",
    repo: "Repozytorium na GitHub",
  },
} as const;

export default pl;
