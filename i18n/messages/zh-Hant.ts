// 繁體中文 (Traditional Chinese)
const zhHant = {
  meta: {
    title: "The CV Comedy Podcast - 用 Gemini 把每份履歷變成一集爆笑節目",
    description:
      "上傳你的履歷，用 Google Gemini 3.5 Flash 搭配多人語音 TTS，生成一集 The CV Comedy Podcast 爆笑節目",
    ogTitle: "The CV Comedy Podcast - 把履歷變成爆笑節目",
    ogDescription:
      "用 Google Gemini 3.5 Flash 與多人語音 TTS，把你的履歷變成一集爆笑節目。",
    keywords: "podcast, 履歷, 喜劇, Google Gemini, TTS, 人工智慧, 幽默",
  },
  header: {
    tagline: "你的履歷就是本集來賓。Gemini 負責寫吐槽稿，還幫忙配音。",
    badges: {
      tts: "✨ 多人語音 TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 高智商幽默",
    },
  },
  theme: {
    toDark: "切換至深色主題",
    toLight: "切換至淺色主題",
    dark: "深色主題",
    light: "淺色主題",
  },
  language: { label: "語言" },
  a11y: {
    step: "步驟 {number}：{title}",
    skip: "跳至主要內容",
  },
  features: {
    toggle: "這是什麼？",
    heading: "把你的履歷變成一集喜劇節目",
    intro:
      "The CV Comedy Podcast 會根據你的履歷生成一集節目，由兩位主持人以高智商幽默點評，腳本與配音一應俱全。一切都在你的瀏覽器裡透過 Google Gemini 處理。",
    items: {
      formats: {
        title: "任何格式",
        desc: "以 PDF、DOCX、TXT 或圖片上傳你的履歷；有需要時，會用 AI 的 OCR 擷取文字。",
      },
      script: {
        title: "AI 腳本",
        desc: "Gemini 會寫出一份尖銳又有趣的腳本，帶著深夜脫口秀的語氣。",
      },
      voices: {
        title: "多重語音",
        desc: "這集節目由兩位聲音各異的主持人配音（multi-speaker TTS）。",
      },
      streaming: {
        title: "邊生成邊收聽",
        desc: "音訊會分段送達：不必等全部完成就能開始收聽。",
      },
      download: {
        title: "下載與分享",
        desc: "把節目匯出成文字、音訊（.wav）或影片，一鍵即可分享。",
      },
      privacy: {
        title: "設計上就重視隱私",
        desc: "你的 API Key 和履歷都直接在你的瀏覽器裡使用，不會經過我們自己的伺服器。",
      },
    },
  },
  apikey: {
    title: "你的 API Key",
    inputLabel: "Google AI API Key",
    placeholder: "在這裡貼上你的 API Key",
    remember: "在此瀏覽器記住",
    getKey: "免費取得 API Key ↗",
    note: "這組 key 會直接從你的瀏覽器呼叫 Google API。若開啟「記住」，只會儲存在這台裝置上。",
  },
  cv: {
    title: "你的履歷",
    dropDrag: "把履歷拖曳到這裡，或點擊選擇檔案",
    dropFormats: "PDF、DOCX、TXT 或圖片（PNG/JPG/WebP）",
    sizeReplace: "{size} KB · 點擊即可更換",
    processing: "正在處理檔案...",
    ocring: "正在用 Gemini（OCR）擷取文件文字...",
    ocrButton: "🔍 用 AI 擷取文字（OCR）",
    textLabel: "履歷文字",
    clearFile: "移除檔案",
    placeholderFile:
      "從你履歷擷取出的文字會顯示在這裡。你可以在生成節目前先編輯...",
    placeholderManual: "...或直接在這裡貼上你的履歷文字",
    errors: {
      reupload: "請重新上傳檔案，才能用 AI 擷取文字。",
      ocrNeedsKey: "請輸入你的 API Key（步驟 1），才能用 OCR（AI）擷取文字。",
      ocrFailed: "無法用 AI 擷取文字：{reason}",
      ocrEmpty: "模型沒有回傳任何文字",
      pdfScanned:
        "無法從 PDF 擷取文字（看起來是掃描檔）。請輸入你的 API Key 並按下「用 AI 擷取文字（OCR）」，或手動貼上文字。",
      imageNeedsOcr:
        "要從圖片讀取履歷需使用 AI 的 OCR。請輸入你的 API Key（步驟 1）並按下「用 AI 擷取文字（OCR）」。",
      docxEmpty: "無法從 DOCX 擷取文字。請手動貼上履歷。",
      txtEmpty: "這個 TXT 檔是空的。請補上內容，或手動貼上履歷。",
      unsupported:
        "不支援的檔案類型。請使用 PDF、DOCX、TXT 或圖片（PNG/JPG/WebP）。",
      processFailed: "處理檔案時發生錯誤：{reason}",
    },
  },
  episode: {
    title: "你的節目",
    generate: "🎭 生成節目",
    regenerate: "🔁 生成一集新節目",
    generating: "正在生成節目...",
    missingKey: "還缺你的 API Key（步驟 1）。",
    missingCv: "還缺你的履歷文字（步驟 2）。",
    waitVideo: "請等影片匯出完成。",
    scriptWriting: "撰寫中...",
    scriptReady: "腳本",
    copy: "📋 複製",
    copied: "✓ 已複製",
    share: "📣 分享",
    linkCopied: "✓ 已複製連結",
    audioFile: "🎵 音訊（.wav）",
    scriptFile: "📄 腳本（.txt）",
    video: "🎬 影片",
    cancelVideo: "✕ 取消影片",
    newEpisode: "✨ 新節目",
    shareText: "來聽聽我的履歷被做成的爆笑節目 🎙️😂",
    progress: {
      video: "正在即時錄製影片（所需時間等同節目長度）...",
      writing: "正在撰寫腳本...",
      writingWith: "正在用 {model} 撰寫腳本...",
      recording: "正在錄製節目...",
      recordingPart: "正在錄製第 {current} / {total} 段...",
      partsReady: "{label}（已完成 {done}/{total} 段）",
      preparingAudio: "正在準備音訊...",
    },
    errors: {
      scriptFailed: "生成節目時發生錯誤：{reason}",
      scriptEmpty: "模型沒有回傳節目腳本",
      audioFailed:
        "生成音訊時發生錯誤：{reason} 已生成的部分會保留：你可以從中斷處繼續。",
      noAudio: "回應中沒有音訊",
      videoFailed: "無法匯出影片：{reason}",
      copyFailed: "無法複製到剪貼簿。",
      resume: "🔁 繼續生成音訊（從第 {part} 段）",
      retryAudio: "🔁 重試音訊",
    },
  },
  api: {
    unavailable:
      "Gemini 模型目前流量壅塞（錯誤 503）。這通常是暫時的：請等幾分鐘後再重試。",
    quota:
      "你已達到 API Key 的配額上限（錯誤 429）。請等一分鐘後再重試，或到 Google AI Studio 查看你的方案。",
    invalidKey: "這組 API Key 無效或沒有權限。請到 Google AI Studio 檢查。",
    unknown: "未知錯誤",
  },
  player: {
    complete: "節目已完成",
    playBlocked: "按下播放以繼續（第 {current} / {total} 段）",
    waiting: "正在等待節目的下一段...",
    part: "第 {current} / {total} 段 · 其餘部分生成時你可以先聽",
    preparing: "正在準備第一段音訊...",
    empty: "節目的音訊會顯示在這裡",
    cover: "節目封面",
    audioLabel: "節目音訊",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ 有讓你笑出來嗎？",
    text: "到 GitHub Sponsors 支持這個專案，讓節目繼續播下去。",
    button: "在 GitHub Sponsors 贊助",
  },
  prompt: {
    script:
      '歡迎收聽 The CV Comedy Podcast。每一份履歷都是全新的一集。你們是一對專業喜劇搭檔，專精於為 podcast 製作高智商的幽默內容。請為一集 4 到 6 分鐘的節目撰寫腳本，用有趣又帶點嘲諷的方式吐槽一份履歷（這份履歷就是本集的來賓）。\n\n多人語音 TTS 的必要格式：\nAlex: [第一位主持人的台詞]\nSam: [第二位主持人的台詞]\n\n主持人特質：\n- Alex：善於分析又愛嘲諷，會做出精準的技術性觀察\n- Sam：即興又逗趣，會丟出好笑的評論和隨興的觀察\n\n要用高智商幽默吐槽的元素：\n- 常見的陳腔濫調："我非常追求完美"、"我很擅長團隊合作"\n- 時間軸或邏輯上的矛盾\n- 誇大的技能："樣樣精通"\n- 把基層工作講得天花亂墜\n- 含糊的職涯目標："我想在職涯上有所成長"\n- 無關緊要或老套的興趣\n- 拼字或文法錯誤\n\n語氣：嘲諷但不失格調，像深夜喜劇脫口秀。保持高智商的幽默，別太刻薄。\n\n重要：\n- 每一句台詞都要精確地使用 "Alex:" 和 "Sam:"\n- 用 "[...]" 加入自然的停頓\n- 在適當的地方用 "[énfasis]" 加上重音\n- 讓對話自然地流動\n- 最多三分半鐘。最多\n\n時間背景：今天是 {date}。請以這個真實日期為基準來評估履歷上的日期：比你所知更新或更晚的日期並不算時間軸矛盾。\n\n請分析這份履歷，並用繁體中文撰寫這集的腳本，而且要非常非常尖銳（根本就是一場毫不留情的吐槽）：',
    vibes:
      '額外素材：附上的是履歷的原始文件。請用喜劇演員的眼光來看它：照片、排版、字體、配色、整體的 "vibe"。如果有哪個視覺元素能拿來開個好笑話，就用它（恰到好處地提個一兩次就好），但別照字面描述它，也別讓它變成整集的焦點。',
    ttsStyle:
      '請用繁體中文生成一集脫口秀喜劇的音訊，形式是吐槽履歷的 podcast，語氣像深夜脫口秀：嘲諷但不失格調、高智商的幽默、別太刻薄。每一句台詞都要精確使用主持人的名字，並在適當處用 "[...]" 加入自然的停頓、用 "[énfasis]" 加上重音。讓對話自然地流動，就像一場深夜喜劇秀。',
    ocr: "請擷取並轉錄這份履歷（CV）中的所有文字。只回傳文件的純文字內容，不要加任何註解或 markdown 格式。請用換行保留原有結構（章節、日期、清單）。",
  },
  footer: {
    disclaimer:
      "⚠️ 本應用以 Vibe Coding（AI）開發，直接從瀏覽器呼叫 Google AI 的 API。",
    repo: "GitHub 儲存庫",
  },
} as const;

export default zhHant;
