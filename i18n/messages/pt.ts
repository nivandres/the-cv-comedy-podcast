// Português
const pt = {
  meta: {
    title:
      "The CV Comedy Podcast - Transforme qualquer currículo em um episódio de comédia com Gemini",
    description:
      "Envie seu currículo e gere um episódio de comédia do The CV Comedy Podcast usando o Google Gemini 3.5 Flash com TTS multi-speaker",
    ogTitle:
      "The CV Comedy Podcast - Transforme currículos em episódios de comédia",
    ogDescription:
      "Transforme seu currículo em um episódio de comédia com Google Gemini 3.5 Flash e Multi-Speaker TTS.",
    keywords:
      "podcast, currículo, CV, comédia, Google Gemini, TTS, inteligência artificial, humor",
  },
  header: {
    tagline:
      "Seu currículo é o convidado. O Gemini escreve o roast e dá as vozes.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Humor inteligente",
    },
  },
  theme: {
    toDark: "Mudar para tema escuro",
    toLight: "Mudar para tema claro",
    dark: "Tema escuro",
    light: "Tema claro",
  },
  language: { label: "Idioma" },
  a11y: {
    step: "Passo {number}: {title}",
    skip: "Pular para o conteúdo",
  },
  features: {
    toggle: "O que é isso?",
    heading: "Transforme seu currículo em um episódio de comédia",
    intro:
      "O The CV Comedy Podcast pega o seu currículo e gera um episódio em que dois apresentadores o comentam com humor inteligente, com roteiro e vozes incluídos. Tudo é processado no seu navegador com o Google Gemini.",
    items: {
      formats: {
        title: "Qualquer formato",
        desc: "Envie seu currículo em PDF, DOCX, TXT ou como imagem; se necessário, o texto é extraído com OCR de IA.",
      },
      script: {
        title: "Roteiro com IA",
        desc: "O Gemini escreve um roteiro crítico e divertido, com o tom de um late-night show.",
      },
      voices: {
        title: "Vozes múltiplas",
        desc: "O episódio é narrado por dois apresentadores com vozes distintas (multi-speaker TTS).",
      },
      streaming: {
        title: "Ouça enquanto é gerado",
        desc: "O áudio chega em partes: você começa a ouvir sem esperar tudo terminar.",
      },
      download: {
        title: "Baixe e compartilhe",
        desc: "Exporte o episódio como texto, áudio (.wav) ou vídeo, e compartilhe com um clique.",
      },
      privacy: {
        title: "Privado por design",
        desc: "Sua API Key e seu currículo são usados diretamente do seu navegador, sem passar por um servidor próprio.",
      },
    },
  },
  apikey: {
    title: "Sua API Key",
    inputLabel: "Google AI API Key",
    placeholder: "Cole sua API Key aqui",
    remember: "Lembrar neste navegador",
    getKey: "Obtenha uma API Key grátis ↗",
    note: "A key é usada diretamente do seu navegador contra a API do Google. Se você ativar «Lembrar», ela fica salva apenas neste dispositivo.",
  },
  cv: {
    title: "Seu currículo",
    dropDrag: "Arraste seu currículo aqui ou clique para selecionar",
    dropFormats: "PDF, DOCX, TXT ou imagem (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · clique para substituí-lo",
    processing: "Processando arquivo...",
    ocring: "Extraindo texto do documento com Gemini (OCR)...",
    ocrButton: "🔍 Extrair texto com IA (OCR)",
    textLabel: "Texto do currículo",
    clearFile: "Remover arquivo",
    placeholderFile:
      "O texto extraído do seu currículo aparecerá aqui. Você pode editá-lo antes de gerar o episódio...",
    placeholderManual: "...ou cole aqui o texto do seu currículo diretamente",
    errors: {
      reupload: "Envie o arquivo novamente para extrair o texto com IA.",
      ocrNeedsKey:
        "Insira sua API Key (passo 1) para extrair o texto com OCR (IA).",
      ocrFailed: "Não foi possível extrair o texto com IA: {reason}",
      ocrEmpty: "O modelo não retornou texto",
      pdfScanned:
        "Não foi possível extrair texto do PDF (parece escaneado). Insira sua API Key e pressione «Extrair texto com IA (OCR)», ou cole o texto manualmente.",
      imageNeedsOcr:
        "Ler um currículo a partir de uma imagem usa OCR com IA. Insira sua API Key (passo 1) e pressione «Extrair texto com IA (OCR)».",
      docxEmpty:
        "Não foi possível extrair texto do DOCX. Cole seu currículo manualmente.",
      txtEmpty:
        "O arquivo TXT está vazio. Adicione conteúdo ou cole seu currículo manualmente.",
      unsupported:
        "Tipo de arquivo não suportado. Use PDF, DOCX, TXT ou uma imagem (PNG/JPG/WebP).",
      processFailed: "Erro ao processar o arquivo: {reason}",
    },
  },
  episode: {
    title: "Seu episódio",
    generate: "🎭 Gerar episódio",
    regenerate: "🔁 Gerar um novo episódio",
    generating: "Gerando episódio...",
    missingKey: "Falta sua API Key (passo 1).",
    missingCv: "Falta o texto do seu currículo (passo 2).",
    waitVideo: "Aguarde a exportação do vídeo terminar.",
    scriptWriting: "Escrevendo...",
    scriptReady: "Roteiro",
    copy: "📋 Copiar",
    copied: "✓ Copiado",
    share: "📣 Compartilhar",
    linkCopied: "✓ Link copiado",
    audioFile: "🎵 Áudio (.wav)",
    scriptFile: "📄 Roteiro (.txt)",
    video: "🎬 Vídeo",
    cancelVideo: "✕ Cancelar vídeo",
    newEpisode: "✨ Novo episódio",
    shareText: "Ouça o episódio de comédia do meu currículo 🎙️😂",
    progress: {
      video: "Gravando o vídeo em tempo real (leva o tempo do episódio)...",
      writing: "Escrevendo o roteiro...",
      writingWith: "Escrevendo o roteiro com {model}...",
      recording: "Gravando o episódio...",
      recordingPart: "Gravando parte {current} de {total}...",
      partsReady: "{label} ({done}/{total} partes prontas)",
      preparingAudio: "Preparando o áudio...",
    },
    errors: {
      scriptFailed: "Erro ao gerar o episódio: {reason}",
      scriptEmpty: "O modelo não retornou o roteiro do episódio",
      audioFailed:
        "Erro ao gerar o áudio: {reason} O que já foi gerado é mantido: você pode retomar de onde parou.",
      noAudio: "A resposta não contém áudio",
      videoFailed: "Não foi possível exportar o vídeo: {reason}",
      copyFailed: "Não foi possível copiar para a área de transferência.",
      resume: "🔁 Retomar áudio (a partir da parte {part})",
      retryAudio: "🔁 Tentar áudio novamente",
    },
  },
  api: {
    unavailable:
      "Os modelos do Gemini estão sobrecarregados no momento (erro 503). Costuma ser temporário: espere alguns minutos e tente novamente.",
    quota:
      "Você atingiu o limite de cota da sua API Key (erro 429). Espere um minuto e tente novamente, ou verifique seu plano no Google AI Studio.",
    invalidKey:
      "A API Key não é válida ou não tem permissões. Verifique-a no Google AI Studio.",
    unknown: "Erro desconhecido",
  },
  player: {
    complete: "Episódio completo",
    playBlocked: "Pressione play para continuar (parte {current} de {total})",
    waiting: "Aguardando a próxima parte do episódio...",
    part: "Parte {current} de {total} · você pode ouvir enquanto o resto é gerado",
    preparing: "Preparando a primeira parte do áudio...",
    empty: "O áudio do episódio aparecerá aqui",
    cover: "Capa do episódio",
    audioLabel: "Áudio do episódio",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Te fez rir?",
    text: "Apoie o projeto no GitHub Sponsors para que novos episódios continuem no ar.",
    button: "Patrocinar no GitHub Sponsors",
  },
  prompt: {
    script:
      'Bem-vindo ao The CV Comedy Podcast. Cada currículo é um novo episódio. Vocês são uma dupla de comediantes profissionais especializados em criar conteúdo de humor inteligente para podcasts. Escreva o roteiro de um episódio de 4-6 minutos que critique de forma engraçada e sarcástica um currículo (o currículo é o convidado do episódio).\n\nFORMATO OBRIGATÓRIO para TTS multi-speaker:\nAlex: [falas do primeiro apresentador]\nSam: [falas do segundo apresentador]\n\nPERSONALIDADES DOS APRESENTADORES:\n- Alex: Analítico e sarcástico, faz observações técnicas precisas\n- Sam: Espontâneo e engraçado, faz comentários divertidos e observações casuais\n\nELEMENTOS PARA CRITICAR COM HUMOR INTELIGENTE:\n- Clichês típicos: "sou muito perfeccionista", "trabalho bem em equipe"\n- Inconsistências temporais ou lógicas\n- Habilidades exageradas: "especialista em tudo"\n- Descrições pomposas de trabalhos básicos\n- Objetivos profissionais vagos: "busco crescer profissionalmente"\n- Hobbies irrelevantes ou clichês\n- Erros de ortografia ou gramática\n\nTOM: Sarcástico mas sofisticado, como um late-night show. Mantenha o humor inteligente e evite ser cruel.\n\nIMPORTANTE:\n- Use EXATAMENTE "Alex:" e "Sam:" para cada fala\n- Inclua pausas naturais com "[...]"\n- Adicione ênfase com "[énfasis]" onde for apropriado\n- Faça a conversa fluir naturalmente\n- Máximo 3 minutos e meio. MÁXIMO\n\nCONTEXTO TEMPORAL: Hoje é {date}. Avalie as datas do currículo em relação a esta data real: uma data recente ou posterior ao seu conhecimento NÃO é uma inconsistência temporal.\n\nAnalise este currículo e escreva o roteiro do episódio em português, e bastante bastante crítico (literalmente um roast sem piedade):',
    vibes:
      "MATERIAL EXTRA: o documento original do currículo está anexado. Olhe para ele com olhar de comediante: a foto, o design, a tipografia, as cores, o vibe geral. Se algo visual render uma boa piada, use (uma ou duas menções bem colocadas), mas não o descreva literalmente nem o transforme no centro do episódio.",
    ttsStyle:
      'Gere o áudio de um episódio de standup comedy em português, no formato de um podcast crítico de currículos, com o tom de um late-night show: sarcástico mas sofisticado, humor inteligente, evite ser cruel. Use exatamente os nomes dos apresentadores para cada fala, inclua pausas naturais com "[...]" e ênfase com "[énfasis]" onde for apropriado. Faça a conversa fluir naturalmente, como um show de comédia noturno.',
    ocr: "Extraia e transcreva TODO o texto deste currículo (CV). Retorne apenas o texto puro do documento, sem comentários nem formatação markdown. Preserve a estrutura (seções, datas, listas) com quebras de linha.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplicativo desenvolvido com Vibe Coding (IA). Usa a API do Google AI diretamente do seu navegador.",
    repo: "Repositório no GitHub",
  },
} as const;

export default pt;
