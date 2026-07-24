// 简体中文 (Simplified Chinese)
const zhHans = {
  meta: {
    title: "The CV Comedy Podcast - 用 Gemini 把每份简历变成一期爆笑节目",
    description:
      "上传你的简历，用 Google Gemini 3.5 Flash 加多人语音 TTS，生成一期 The CV Comedy Podcast 爆笑节目",
    ogTitle: "The CV Comedy Podcast - 把简历变成爆笑播客节目",
    ogDescription:
      "用 Google Gemini 3.5 Flash 加多人语音 TTS，把你的简历变成一期爆笑节目。",
    keywords: "播客, 简历, 喜剧, Google Gemini, TTS, 人工智能, 幽默",
  },
  header: {
    tagline: "你的简历就是本期嘉宾。Gemini 负责写吐槽稿，还顺便给它配音。",
    badges: {
      tts: "✨ 多人语音 TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 高级幽默",
    },
  },
  theme: {
    toDark: "切换到深色主题",
    toLight: "切换到浅色主题",
    dark: "深色主题",
    light: "浅色主题",
  },
  language: { label: "语言" },
  a11y: {
    step: "第 {number} 步：{title}",
    skip: "跳到主要内容",
  },
  features: {
    toggle: "这是什么？",
    heading: "把你的简历变成一期喜剧节目",
    intro:
      "The CV Comedy Podcast 会根据你的简历生成一期节目：由两位主持人用高级幽默来点评，台词稿和配音一应俱全。整个过程都在你的浏览器里用 Google Gemini 完成。",
    items: {
      formats: {
        title: "任何格式都行",
        desc: "上传 PDF、DOCX、TXT 或图片格式的简历；必要时会用 AI OCR 提取其中的文字。",
      },
      script: {
        title: "AI 撰写台词稿",
        desc: "Gemini 会写出一份既犀利又好笑的台词稿，带着深夜脱口秀的语气。",
      },
      voices: {
        title: "多人配音",
        desc: "节目由两位嗓音不同的主持人播讲（multi-speaker TTS）。",
      },
      streaming: {
        title: "边生成边收听",
        desc: "音频会分段送达：无需等到全部完成，就能开始收听。",
      },
      download: {
        title: "下载并分享",
        desc: "把节目导出为文字、音频（.wav）或视频，一键分享。",
      },
      privacy: {
        title: "从设计上保护隐私",
        desc: "你的 API Key 和简历都直接从你的浏览器使用，不会经过我们自己的服务器。",
      },
    },
  },
  apikey: {
    title: "你的 API Key",
    inputLabel: "Google AI API Key",
    placeholder: "在此粘贴你的 API Key",
    remember: "在此浏览器中记住",
    getKey: "免费获取 API Key ↗",
    note: "该密钥会直接从你的浏览器调用 Google 的 API。如果开启「记住」，它只会保存在本设备上。",
  },
  cv: {
    title: "你的简历",
    dropDrag: "把简历拖到这里，或点击选择文件",
    dropFormats: "PDF、DOCX、TXT 或图片（PNG/JPG/WebP）",
    sizeReplace: "{size} KB · 点击可替换",
    processing: "正在处理文件…",
    ocring: "正在用 Gemini 从文档中提取文字（OCR）…",
    ocrButton: "🔍 用 AI 提取文字（OCR）",
    textLabel: "简历文字",
    clearFile: "移除文件",
    placeholderFile:
      "从简历中提取的文字会显示在这里。你可以在生成节目前先编辑…",
    placeholderManual: "…或者直接把简历文字粘贴到这里",
    errors: {
      reupload: "请重新上传文件，以便用 AI 提取文字。",
      ocrNeedsKey:
        "请先输入你的 API Key（第 1 步），才能用 OCR（AI）提取文字。",
      ocrFailed: "无法用 AI 提取文字：{reason}",
      ocrEmpty: "模型没有返回任何文字",
      pdfScanned:
        "无法从 PDF 中提取文字（看起来是扫描件）。请输入你的 API Key 并点击「用 AI 提取文字（OCR）」，或手动粘贴文字。",
      imageNeedsOcr:
        "要从图片中读取简历，需要用 AI OCR。请输入你的 API Key（第 1 步）并点击「用 AI 提取文字（OCR）」。",
      docxEmpty: "无法从 DOCX 中提取文字。请手动粘贴简历。",
      txtEmpty: "TXT 文件是空的。请添加内容，或手动粘贴简历。",
      unsupported:
        "不支持的文件类型。请使用 PDF、DOCX、TXT 或图片（PNG/JPG/WebP）。",
      processFailed: "处理文件时出错：{reason}",
    },
  },
  episode: {
    title: "你的节目",
    generate: "🎭 生成节目",
    regenerate: "🔁 重新生成一期节目",
    generating: "正在生成节目…",
    missingKey: "缺少你的 API Key（第 1 步）。",
    missingCv: "缺少你的简历文字（第 2 步）。",
    waitVideo: "请等待视频导出完成。",
    scriptWriting: "正在撰写…",
    scriptReady: "台词稿",
    copy: "📋 复制",
    copied: "✓ 已复制",
    share: "📣 分享",
    linkCopied: "✓ 链接已复制",
    audioFile: "🎵 音频（.wav）",
    scriptFile: "📄 台词稿（.txt）",
    video: "🎬 视频",
    cancelVideo: "✕ 取消视频",
    newEpisode: "✨ 新节目",
    shareText: "来听听我简历的爆笑吐槽节目 🎙️😂",
    progress: {
      video: "正在实时录制视频（耗时和节目时长一样）…",
      writing: "正在撰写台词稿…",
      writingWith: "正在用 {model} 撰写台词稿…",
      recording: "正在录制节目…",
      recordingPart: "正在录制第 {current} / {total} 部分…",
      partsReady: "{label}（已完成 {done}/{total} 部分）",
      preparingAudio: "正在准备音频…",
    },
    errors: {
      scriptFailed: "生成节目时出错：{reason}",
      scriptEmpty: "模型没有返回节目台词稿",
      audioFailed:
        "生成音频时出错：{reason} 已生成的内容会保留：你可以从中断处继续。",
      noAudio: "响应中不包含音频",
      videoFailed: "无法导出视频：{reason}",
      copyFailed: "无法复制到剪贴板。",
      resume: "🔁 继续生成音频（从第 {part} 部分开始）",
      retryAudio: "🔁 重试音频",
    },
  },
  api: {
    unavailable:
      "Gemini 模型目前负载过高（错误 503）。通常只是暂时的：请等待几分钟后重试。",
    quota:
      "你的 API Key 已达到配额上限（错误 429）。请等待一分钟后重试，或在 Google AI Studio 中查看你的套餐。",
    invalidKey: "API Key 无效或没有权限。请在 Google AI Studio 中检查。",
    unknown: "未知错误",
  },
  player: {
    complete: "节目已完成",
    playBlocked: "点击播放以继续（第 {current} / {total} 部分）",
    waiting: "正在等待节目的下一部分…",
    part: "第 {current} / {total} 部分 · 你可以一边听，一边等其余部分生成",
    preparing: "正在准备音频的第一部分…",
    empty: "节目音频会显示在这里",
    cover: "节目封面",
    audioLabel: "节目音频",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ 有没有把你逗笑？",
    text: "在 GitHub Sponsors 上支持这个项目，让节目继续上线播出。",
    button: "在 GitHub Sponsors 上赞助",
  },
  prompt: {
    script:
      '欢迎收听 The CV Comedy Podcast。每一份简历都是一期新节目。你们是一对专业喜剧搭档，专门为播客创作既聪明又幽默的内容。请撰写一期 4 至 6 分钟的节目台词稿，用有趣又辛辣的方式吐槽一份简历（这份简历就是本期嘉宾）。\n\n多人语音 TTS 所需格式：\nAlex: [第一位主持人的台词]\nSam: [第二位主持人的台词]\n\n两位主持人的性格设定：\n- Alex：擅长分析、语带讽刺，会给出精准的技术性点评\n- Sam：随性又搞笑，负责抛出好玩的段子和随口的吐槽\n\n用高级幽默狠狠吐槽这些点：\n- 老掉牙的套话："我非常追求完美"、"我很擅长团队合作"\n- 时间线或逻辑上自相矛盾的地方\n- 夸大其词的技能："样样精通"\n- 把最基础的工作吹得天花乱坠\n- 含糊其辞的职业目标："希望在职业上不断成长"\n- 不相干或老套的兴趣爱好\n- 拼写或语法错误\n\n语气：讽刺但不失格调，像深夜喜剧脱口秀。保持高级幽默，别刻薄伤人。\n\n重要事项：\n- 每段台词都必须精确地用 "Alex:" 和 "Sam:" 开头\n- 用 "[...]" 加入自然的停顿\n- 在合适的地方用 "[énfasis]" 加以强调\n- 让对话自然流畅\n- 最长三分半钟。最长\n\n时间背景：今天是 {date}。请以这个真实日期为基准来评估简历上的日期：一个较近、或晚于你知识截止时间的日期，并不算时间线矛盾。\n\n分析这份简历，用简体中文撰写本期节目的台词稿，而且要相当相当犀利（简直就是一场毫不留情的吐槽大会）：',
    vibes:
      '附加素材：随附的是简历的原始文档。请用喜剧演员的眼光来看它：照片、版式、字体、配色，还有整体的"氛围感"。如果某个视觉元素能抖个好包袱，就用上它（恰到好处地提一两次），但不要照字面描述它，也别让它成为整期节目的焦点。',
    ttsStyle:
      '请用简体中文生成一期脱口秀喜剧节目的音频，形式为吐槽简历的播客，语气像深夜脱口秀：讽刺但不失格调，高级幽默，别刻薄伤人。每段台词都精确使用主持人的名字，在合适的地方用 "[...]" 表示自然停顿、用 "[énfasis]" 表示强调。让对话自然流畅，就像一场深夜喜剧秀。',
    ocr: "提取并转录这份简历（履历）中的全部文字。只返回文档的纯文本，不要任何评论，也不要 markdown 格式。请用换行保留原有结构（章节、日期、列表）。",
  },
  footer: {
    disclaimer:
      "⚠️ 本应用使用 Vibe Coding（AI）开发，直接从浏览器调用 Google AI 的 API。",
    repo: "GitHub 仓库",
  },
} as const;

export default zhHans;
