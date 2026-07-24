// 한국어 (Korean)
const ko = {
  meta: {
    title:
      "The CV Comedy Podcast - Gemini로 모든 이력서를 코미디 에피소드로 바꿔보세요",
    description:
      "이력서를 업로드하고 Google Gemini 3.5 Flash와 멀티 스피커 TTS로 The CV Comedy Podcast의 코미디 에피소드를 만들어보세요",
    ogTitle: "The CV Comedy Podcast - 이력서를 코미디 에피소드로 바꿔보세요",
    ogDescription:
      "Google Gemini 3.5 Flash와 멀티 스피커 TTS로 당신의 이력서를 코미디 에피소드로 바꿔보세요.",
    keywords: "팟캐스트, 이력서, 코미디, Google Gemini, TTS, 인공지능, 유머",
  },
  header: {
    tagline:
      "당신의 이력서가 오늘의 게스트. Gemini가 로스팅 대본을 쓰고 목소리까지 입힙니다.",
    badges: {
      tts: "✨ 멀티 스피커 TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 위트 있는 유머",
    },
  },
  theme: {
    toDark: "다크 모드로 전환",
    toLight: "라이트 모드로 전환",
    dark: "다크 모드",
    light: "라이트 모드",
  },
  language: { label: "언어" },
  a11y: {
    step: "{number}단계: {title}",
    skip: "본문 바로가기",
  },
  features: {
    toggle: "이게 뭔가요?",
    heading: "이력서를 코미디 에피소드로 바꿔보세요",
    intro:
      "The CV Comedy Podcast는 당신의 이력서를 받아 두 진행자가 위트 있는 유머로 이야기를 나누는 에피소드를 만듭니다. 대본과 목소리까지 모두 포함됩니다. 모든 과정은 Google Gemini를 사용해 브라우저에서 처리됩니다.",
    items: {
      formats: {
        title: "어떤 형식이든",
        desc: "PDF, DOCX, TXT 또는 이미지로 이력서를 업로드하세요. 필요하면 AI OCR로 텍스트를 추출합니다.",
      },
      script: {
        title: "AI 대본",
        desc: "Gemini가 심야 쇼 톤으로 신랄하면서도 재미있는 대본을 씁니다.",
      },
      voices: {
        title: "여러 목소리",
        desc: "서로 다른 목소리를 가진 두 진행자가 에피소드를 진행합니다 (multi-speaker TTS).",
      },
      streaming: {
        title: "생성되는 동안 바로 듣기",
        desc: "오디오가 파트별로 전달되어, 전체가 끝날 때까지 기다리지 않고 바로 듣기 시작할 수 있습니다.",
      },
      download: {
        title: "다운로드하고 공유하기",
        desc: "에피소드를 텍스트, 오디오(.wav), 동영상으로 내보내고 클릭 한 번으로 공유하세요.",
      },
      privacy: {
        title: "기본이 프라이버시",
        desc: "API 키와 이력서는 브라우저에서 바로 사용되며, 자체 서버를 거치지 않습니다.",
      },
    },
  },
  apikey: {
    title: "API 키",
    inputLabel: "Google AI API 키",
    placeholder: "여기에 API 키를 붙여넣으세요",
    remember: "이 브라우저에 기억하기",
    getKey: "무료 API 키 발급받기 ↗",
    note: "키는 브라우저에서 Google API로 바로 전송됩니다. '기억하기'를 켜면 이 기기에만 저장됩니다.",
  },
  cv: {
    title: "이력서",
    dropDrag: "이력서를 여기에 끌어다 놓거나 클릭해서 선택하세요",
    dropFormats: "PDF, DOCX, TXT 또는 이미지 (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · 클릭해서 교체하기",
    processing: "파일 처리 중...",
    ocring: "Gemini로 문서에서 텍스트를 추출하는 중 (OCR)...",
    ocrButton: "🔍 AI로 텍스트 추출 (OCR)",
    textLabel: "이력서 텍스트",
    clearFile: "파일 제거",
    placeholderFile:
      "이력서에서 추출한 텍스트가 여기에 표시됩니다. 에피소드를 만들기 전에 편집할 수 있어요...",
    placeholderManual: "...또는 이력서 텍스트를 여기에 바로 붙여넣으세요",
    errors: {
      reupload: "AI로 텍스트를 추출하려면 파일을 다시 업로드하세요.",
      ocrNeedsKey: "OCR(AI)로 텍스트를 추출하려면 API 키를 입력하세요 (1단계).",
      ocrFailed: "AI로 텍스트를 추출하지 못했습니다: {reason}",
      ocrEmpty: "모델이 텍스트를 반환하지 않았습니다",
      pdfScanned:
        "PDF에서 텍스트를 추출하지 못했습니다 (스캔본으로 보입니다). API 키를 입력하고 'AI로 텍스트 추출 (OCR)'을 누르거나 텍스트를 직접 붙여넣으세요.",
      imageNeedsOcr:
        "이미지에서 이력서를 읽으려면 AI OCR을 사용합니다. API 키를 입력하고 (1단계) 'AI로 텍스트 추출 (OCR)'을 누르세요.",
      docxEmpty:
        "DOCX에서 텍스트를 추출하지 못했습니다. 이력서를 직접 붙여넣으세요.",
      txtEmpty:
        "TXT 파일이 비어 있습니다. 내용을 추가하거나 이력서를 직접 붙여넣으세요.",
      unsupported:
        "지원하지 않는 파일 형식입니다. PDF, DOCX, TXT 또는 이미지 (PNG/JPG/WebP)를 사용하세요.",
      processFailed: "파일 처리 오류: {reason}",
    },
  },
  episode: {
    title: "에피소드",
    generate: "🎭 에피소드 생성",
    regenerate: "🔁 새 에피소드 생성",
    generating: "에피소드 생성 중...",
    missingKey: "API 키가 없습니다 (1단계).",
    missingCv: "이력서 텍스트가 없습니다 (2단계).",
    waitVideo: "동영상 내보내기가 끝날 때까지 기다려 주세요.",
    scriptWriting: "작성 중...",
    scriptReady: "대본",
    copy: "📋 복사",
    copied: "✓ 복사됨",
    share: "📣 공유",
    linkCopied: "✓ 링크 복사됨",
    audioFile: "🎵 오디오 (.wav)",
    scriptFile: "📄 대본 (.txt)",
    video: "🎬 동영상",
    cancelVideo: "✕ 동영상 취소",
    newEpisode: "✨ 새 에피소드",
    shareText: "제 이력서로 만든 코미디 에피소드를 들어보세요 🎙️😂",
    progress: {
      video: "실시간으로 동영상을 녹화하는 중 (에피소드 길이만큼 걸립니다)...",
      writing: "대본을 작성하는 중...",
      writingWith: "{model}로 대본을 작성하는 중...",
      recording: "에피소드를 녹음하는 중...",
      recordingPart: "{total}개 중 {current}번째 파트를 녹음하는 중...",
      partsReady: "{label} ({total}개 중 {done}개 파트 준비 완료)",
      preparingAudio: "오디오를 준비하는 중...",
    },
    errors: {
      scriptFailed: "에피소드 생성 오류: {reason}",
      scriptEmpty: "모델이 에피소드 대본을 반환하지 않았습니다",
      audioFailed:
        "오디오 생성 오류: {reason} 이미 생성된 부분은 그대로 보존됩니다. 중단된 지점부터 이어서 진행할 수 있어요.",
      noAudio: "응답에 오디오가 없습니다",
      videoFailed: "동영상을 내보내지 못했습니다: {reason}",
      copyFailed: "클립보드에 복사하지 못했습니다.",
      resume: "🔁 오디오 이어서 진행 ({part}번째 파트부터)",
      retryAudio: "🔁 오디오 다시 시도",
    },
  },
  api: {
    unavailable:
      "지금 Gemini 모델에 요청이 몰려 있습니다 (503 오류). 대개 일시적이니 몇 분 후에 다시 시도해 주세요.",
    quota:
      "API 키의 할당량 한도에 도달했습니다 (429 오류). 잠시 후 다시 시도하거나 Google AI Studio에서 요금제를 확인하세요.",
    invalidKey:
      "API 키가 유효하지 않거나 권한이 없습니다. Google AI Studio에서 확인하세요.",
    unknown: "알 수 없는 오류",
  },
  player: {
    complete: "에피소드 완성",
    playBlocked: "계속하려면 재생을 누르세요 ({total}개 중 {current}번째 파트)",
    waiting: "에피소드의 다음 파트를 기다리는 중...",
    part: "{total}개 중 {current}번째 파트 · 나머지가 생성되는 동안 들을 수 있어요",
    preparing: "오디오의 첫 파트를 준비하는 중...",
    empty: "에피소드 오디오가 여기에 표시됩니다",
    cover: "에피소드 커버",
    audioLabel: "에피소드 오디오",
    cc: "💬 자막",
  },
  sponsor: {
    title: "❤️ 웃음이 나셨나요?",
    text: "더 많은 에피소드가 계속 나올 수 있도록 GitHub Sponsors에서 프로젝트를 후원해 주세요.",
    button: "GitHub Sponsors에서 후원하기",
  },
  prompt: {
    script:
      'The CV Comedy Podcast에 오신 것을 환영합니다. 모든 이력서는 새로운 에피소드입니다. 당신은 팟캐스트를 위한 위트 있는 코미디 콘텐츠 제작에 특화된 프로 코미디언 듀오입니다. 이력서를 재치 있고 신랄하게 비평하는 4~6분 분량의 에피소드 대본을 작성하세요 (이력서가 이 에피소드의 게스트입니다).\n\n멀티 스피커 TTS를 위한 필수 형식:\nAlex: [첫 번째 진행자의 대사]\nSam: [두 번째 진행자의 대사]\n\n진행자 특징:\n- Alex: 분석적이고 냉소적이며, 정확한 기술적 관찰을 합니다\n- Sam: 즉흥적이고 유쾌하며, 재미있는 코멘트와 가벼운 관찰을 합니다\n\n위트 있는 유머로 비평할 요소:\n- 흔한 클리셰: "저는 완벽주의자예요", "팀워크가 좋습니다"\n- 시간적 또는 논리적 모순\n- 과장된 능력: "모든 것에 전문가"\n- 평범한 업무에 대한 거창한 설명\n- 모호한 커리어 목표: "전문적으로 성장하고 싶습니다"\n- 관련 없거나 뻔한 취미\n- 맞춤법이나 문법 오류\n\n톤: 냉소적이지만 세련되게, 심야 코미디 쇼처럼. 위트 있는 유머를 유지하되 잔인해지지는 마세요.\n\n중요:\n- 각 대사마다 반드시 "Alex:"와 "Sam:"을 정확히 사용하세요\n- "[...]"로 자연스러운 멈춤을 넣으세요\n- 적절한 곳에 "[énfasis]"로 강조를 더하세요\n- 대화가 자연스럽게 흘러가게 하세요\n- 최대 3분 30초. 최대\n\n시간적 맥락: 오늘은 {date}입니다. 이력서의 날짜를 이 실제 날짜를 기준으로 평가하세요. 최근이거나 당신의 지식 이후의 날짜는 시간적 모순이 아닙니다.\n\n이 이력서를 분석하고 에피소드 대본을 한국어로, 아주아주 신랄하게 작성하세요 (말 그대로 자비 없는 로스트):',
    vibes:
      'MATERIAL EXTRA: 이력서 원본 문서를 함께 첨부합니다. 코미디언의 눈으로 살펴보세요. 사진, 디자인, 글꼴, 색상, 전반적인 "분위기"까지. 시각적인 요소 중 좋은 농담거리가 있다면 활용하되 (잘 배치한 한두 번의 언급 정도), 그대로 묘사하거나 에피소드의 중심으로 삼지는 마세요.',
    ttsStyle:
      '이력서를 비평하는 팟캐스트 형식의 스탠드업 코미디 에피소드 오디오를 한국어로 생성하세요. 심야 쇼의 톤으로: 냉소적이지만 세련되게, 위트 있는 유머로, 잔인해지지는 마세요. 각 대사마다 진행자의 이름을 정확히 사용하고, 적절한 곳에 "[...]"로 자연스러운 멈춤과 "[énfasis]"로 강조를 넣으세요. 대화가 심야 코미디 쇼처럼 자연스럽게 흘러가게 하세요.',
    ocr: "이 이력서(CV)의 모든 텍스트를 추출하고 그대로 옮겨 적으세요. 문서의 순수 텍스트만 반환하고, 코멘트나 마크다운 서식은 넣지 마세요. 구조(섹션, 날짜, 목록)는 줄바꿈으로 유지하세요.",
  },
  footer: {
    disclaimer:
      "⚠️ Vibe Coding (AI)으로 개발한 애플리케이션입니다. 브라우저에서 직접 Google AI API를 사용합니다.",
    repo: "GitHub 저장소",
  },
} as const;

export default ko;
