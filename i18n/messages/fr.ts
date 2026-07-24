// Français
const fr = {
  meta: {
    title:
      "The CV Comedy Podcast - Transformez n'importe quel CV en épisode comique avec Gemini",
    description:
      "Téléversez votre CV et générez un épisode comique de The CV Comedy Podcast avec Google Gemini 3.5 Flash et le TTS multi-voix",
    ogTitle: "The CV Comedy Podcast - Transformez les CV en épisodes comiques",
    ogDescription:
      "Transformez votre CV en épisode comique avec Google Gemini 3.5 Flash et le Multi-Speaker TTS.",
    keywords:
      "podcast, CV, comédie, Google Gemini, TTS, intelligence artificielle, humour",
  },
  header: {
    tagline:
      "Votre CV est l'invité. Gemini écrit le roast et lui prête sa voix.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Humour intelligent",
    },
  },
  theme: {
    toDark: "Passer au thème sombre",
    toLight: "Passer au thème clair",
    dark: "Thème sombre",
    light: "Thème clair",
  },
  language: { label: "Langue" },
  a11y: {
    step: "Étape {number} : {title}",
    skip: "Aller au contenu",
  },
  features: {
    toggle: "Qu'est-ce que c'est ?",
    heading: "Transformez votre CV en épisode comique",
    intro:
      "The CV Comedy Podcast prend votre CV et génère un épisode où deux animateurs le commentent avec un humour intelligent, script et voix inclus. Tout est traité dans votre navigateur avec Google Gemini.",
    items: {
      formats: {
        title: "N'importe quel format",
        desc: "Téléversez votre CV en PDF, DOCX, TXT ou en image ; si nécessaire, le texte est extrait par OCR avec l'IA.",
      },
      script: {
        title: "Script par IA",
        desc: "Gemini écrit un script critique et drôle, avec le ton d'un late-night show.",
      },
      voices: {
        title: "Voix multiples",
        desc: "L'épisode est interprété par deux animateurs aux voix distinctes (multi-speaker TTS).",
      },
      streaming: {
        title: "Écoutez pendant la génération",
        desc: "L'audio arrive par parties : commencez à écouter sans attendre que tout soit terminé.",
      },
      download: {
        title: "Téléchargez et partagez",
        desc: "Exportez l'épisode en texte, audio (.wav) ou vidéo, et partagez-le en un clic.",
      },
      privacy: {
        title: "Privé par conception",
        desc: "Votre clé API et votre CV sont utilisés directement depuis votre navigateur, sans passer par notre propre serveur.",
      },
    },
  },
  apikey: {
    title: "Votre clé API",
    inputLabel: "Clé API Google AI",
    placeholder: "Collez votre clé API ici",
    remember: "Se souvenir sur ce navigateur",
    getKey: "Obtenez une clé API gratuite ↗",
    note: "La clé est utilisée directement depuis votre navigateur vers l'API Google. Si vous activez « Se souvenir », elle n'est enregistrée que sur cet appareil.",
  },
  cv: {
    title: "Votre CV",
    dropDrag: "Glissez votre CV ici ou cliquez pour sélectionner",
    dropFormats: "PDF, DOCX, TXT ou image (PNG/JPG/WebP)",
    sizeReplace: "{size} Ko · cliquez pour le remplacer",
    processing: "Traitement du fichier...",
    ocring: "Extraction du texte du document avec Gemini (OCR)...",
    ocrButton: "🔍 Extraire le texte avec l'IA (OCR)",
    textLabel: "Texte du CV",
    clearFile: "Retirer le fichier",
    placeholderFile:
      "Le texte extrait de votre CV apparaîtra ici. Vous pouvez le modifier avant de générer l'épisode...",
    placeholderManual: "...ou collez directement le texte de votre CV ici",
    errors: {
      reupload:
        "Téléversez à nouveau le fichier pour extraire le texte avec l'IA.",
      ocrNeedsKey:
        "Saisissez votre clé API (étape 1) pour extraire le texte avec l'OCR (IA).",
      ocrFailed: "Impossible d'extraire le texte avec l'IA : {reason}",
      ocrEmpty: "Le modèle n'a renvoyé aucun texte",
      pdfScanned:
        "Impossible d'extraire le texte du PDF (il semble scanné). Saisissez votre clé API et appuyez sur « Extraire le texte avec l'IA (OCR) », ou collez le texte manuellement.",
      imageNeedsOcr:
        "Lire un CV depuis une image utilise l'OCR par IA. Saisissez votre clé API (étape 1) et appuyez sur « Extraire le texte avec l'IA (OCR) ».",
      docxEmpty:
        "Impossible d'extraire le texte du DOCX. Collez votre CV manuellement.",
      txtEmpty:
        "Le fichier TXT est vide. Ajoutez du contenu ou collez votre CV manuellement.",
      unsupported:
        "Type de fichier non pris en charge. Utilisez PDF, DOCX, TXT ou une image (PNG/JPG/WebP).",
      processFailed: "Erreur lors du traitement du fichier : {reason}",
    },
  },
  episode: {
    title: "Votre épisode",
    generate: "🎭 Générer l'épisode",
    regenerate: "🔁 Générer un nouvel épisode",
    generating: "Génération de l'épisode...",
    missingKey: "Votre clé API est manquante (étape 1).",
    missingCv: "Le texte de votre CV est manquant (étape 2).",
    waitVideo: "Attendez la fin de l'export de la vidéo.",
    scriptWriting: "Écriture...",
    scriptReady: "Script",
    copy: "📋 Copier",
    copied: "✓ Copié",
    share: "📣 Partager",
    linkCopied: "✓ Lien copié",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Script (.txt)",
    video: "🎬 Vidéo",
    cancelVideo: "✕ Annuler la vidéo",
    newEpisode: "✨ Nouvel épisode",
    shareText: "Écoutez l'épisode comique de mon CV 🎙️😂",
    progress: {
      video:
        "Enregistrement de la vidéo en temps réel (dure aussi longtemps que l'épisode)...",
      writing: "Écriture du script...",
      writingWith: "Écriture du script avec {model}...",
      recording: "Enregistrement de l'épisode...",
      recordingPart: "Enregistrement de la partie {current} sur {total}...",
      partsReady: "{label} ({done}/{total} parties prêtes)",
      preparingAudio: "Préparation de l'audio...",
    },
    errors: {
      scriptFailed: "Erreur lors de la génération de l'épisode : {reason}",
      scriptEmpty: "Le modèle n'a pas renvoyé le script de l'épisode",
      audioFailed:
        "Erreur lors de la génération de l'audio : {reason} Ce qui a déjà été généré est conservé : vous pouvez reprendre là où ça s'est arrêté.",
      noAudio: "La réponse ne contient pas d'audio",
      videoFailed: "Impossible d'exporter la vidéo : {reason}",
      copyFailed: "Impossible de copier dans le presse-papiers.",
      resume: "🔁 Reprendre l'audio (à partir de la partie {part})",
      retryAudio: "🔁 Réessayer l'audio",
    },
  },
  api: {
    unavailable:
      "Les modèles Gemini sont actuellement saturés (erreur 503). C'est généralement temporaire : attendez quelques minutes et réessayez.",
    quota:
      "Vous avez atteint la limite de quota de votre clé API (erreur 429). Attendez une minute et réessayez, ou vérifiez votre plan dans Google AI Studio.",
    invalidKey:
      "La clé API n'est pas valide ou n'a pas les autorisations. Vérifiez-la dans Google AI Studio.",
    unknown: "Erreur inconnue",
  },
  player: {
    complete: "Épisode complet",
    playBlocked:
      "Appuyez sur play pour continuer (partie {current} sur {total})",
    waiting: "En attente de la prochaine partie de l'épisode...",
    part: "Partie {current} sur {total} · vous pouvez écouter pendant que le reste est généré",
    preparing: "Préparation de la première partie de l'audio...",
    empty: "L'audio de l'épisode apparaîtra ici",
    cover: "Pochette de l'épisode",
    audioLabel: "Audio de l'épisode",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Ça vous a fait rire ?",
    text: "Soutenez le projet sur GitHub Sponsors pour que de nouveaux épisodes continuent d'être diffusés.",
    button: "Parrainer sur GitHub Sponsors",
  },
  prompt: {
    script:
      "Bienvenue à The CV Comedy Podcast. Chaque CV est un nouvel épisode. Vous êtes un duo de comédiens professionnels spécialisés dans la création de contenu humoristique intelligent pour podcasts. Écrivez le script d'un épisode de 4-6 minutes qui critique de façon drôle et sarcastique un CV (le CV est l'invité de l'épisode).\n\nFORMAT REQUIS pour le TTS multi-voix :\nAlex: [répliques du premier animateur]\nSam: [répliques du second animateur]\n\nPERSONNALITÉS DES ANIMATEURS :\n- Alex: Analytique et sarcastique, fait des observations techniques précises\n- Sam: Spontané et drôle, fait des commentaires amusants et des observations décontractées\n\nÉLÉMENTS À CRITIQUER AVEC UN HUMOUR INTELLIGENT :\n- Clichés typiques : « je suis perfectionniste », « j'ai l'esprit d'équipe »\n- Incohérences temporelles ou logiques\n- Compétences exagérées : « expert en tout »\n- Descriptions pompeuses de postes basiques\n- Objectifs professionnels vagues : « je cherche à évoluer professionnellement »\n- Loisirs sans intérêt ou clichés\n- Fautes d'orthographe ou de grammaire\n\nTON : Sarcastique mais sophistiqué, comme un late-night show. Gardez un humour intelligent et évitez d'être cruel.\n\nIMPORTANT :\n- Utilisez EXACTEMENT \"Alex:\" et \"Sam:\" pour chaque réplique\n- Incluez des pauses naturelles avec \"[...]\"\n- Ajoutez de l'emphase avec \"[énfasis]\" quand c'est approprié\n- Faites en sorte que la conversation coule naturellement\n- Maximum 3 minutes et demie. MAXIMUM\n\nCONTEXTE TEMPOREL : Nous sommes le {date}. Évaluez les dates du CV par rapport à cette date réelle : une date récente ou postérieure à vos connaissances n'est PAS une incohérence temporelle.\n\nAnalysez ce CV et écrivez le script de l'épisode en français, et soyez très très critique (littéralement un roast sans pitié) :",
    vibes:
      "MATÉRIEL SUPPLÉMENTAIRE : le document original du CV est joint. Regardez-le avec un œil de comédien : la photo, le design, la typographie, les couleurs, le vibe général. Si un élément visuel se prête à une bonne blague, utilisez-le (une ou deux mentions bien placées), mais ne le décrivez pas littéralement et n'en faites pas le centre de l'épisode.",
    ttsStyle:
      "Générez l'audio d'un épisode de standup comedy en français, au format d'un podcast qui critique des CV, avec le ton d'un late-night show : sarcastique mais sophistiqué, humour intelligent, évitez d'être cruel. Utilisez exactement les noms des animateurs pour chaque réplique, incluez des pauses naturelles avec \"[...]\" et de l'emphase avec \"[énfasis]\" quand c'est approprié. Faites couler la conversation naturellement, comme un show comique nocturne.",
    ocr: "Extrayez et transcrivez TOUT le texte de ce CV. Renvoyez uniquement le texte brut du document, sans commentaires ni formatage markdown. Conservez la structure (sections, dates, listes) avec des sauts de ligne.",
  },
  footer: {
    disclaimer:
      "⚠️ Application développée en Vibe Coding (IA). Elle utilise l'API Google AI directement depuis votre navigateur.",
    repo: "Dépôt GitHub",
  },
} as const;

export default fr;
