// 日本語 (Japanese)
const ja = {
  meta: {
    title:
      "The CV Comedy Podcast - どんな履歴書もGeminiでお笑いエピソードに変換",
    description:
      "履歴書をアップロードして、Google Gemini 3.5 FlashのマルチスピーカーTTSでThe CV Comedy Podcastのお笑いエピソードを生成しよう",
    ogTitle: "The CV Comedy Podcast - 履歴書をお笑いエピソードに変換",
    ogDescription:
      "Google Gemini 3.5 FlashとマルチスピーカーTTSで、あなたの履歴書をお笑いエピソードに変換。",
    keywords:
      "ポッドキャスト, 履歴書, お笑い, Google Gemini, TTS, 人工知能, ユーモア",
  },
  header: {
    tagline: "主役はあなたの履歴書。Geminiがネタを書いて、声まで吹き込みます。",
    badges: {
      tts: "✨ マルチスピーカーTTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 知的なユーモア",
    },
  },
  theme: {
    toDark: "ダークテーマに切り替え",
    toLight: "ライトテーマに切り替え",
    dark: "ダークテーマ",
    light: "ライトテーマ",
  },
  language: { label: "言語" },
  a11y: {
    step: "ステップ{number}: {title}",
    skip: "コンテンツにスキップ",
  },
  features: {
    toggle: "これは何？",
    heading: "あなたの履歴書をお笑いエピソードに変換",
    intro:
      "The CV Comedy Podcastは、あなたの履歴書をもとに、2人の司会者が知的なユーモアで語り合うエピソードを、台本も音声も込みで生成します。処理はすべてGoogle Geminiを使ってブラウザ内で行われます。",
    items: {
      formats: {
        title: "どんな形式でも",
        desc: "履歴書をPDF、DOCX、TXT、または画像でアップロード。必要ならAIのOCRでテキストを抽出します。",
      },
      script: {
        title: "AIによる台本",
        desc: "Geminiが、深夜番組のトーンで辛口かつ愉快な台本を書き上げます。",
      },
      voices: {
        title: "複数の声",
        desc: "エピソードは、声の異なる2人の司会者で読み上げられます（multi-speaker TTS）。",
      },
      streaming: {
        title: "生成しながら聴ける",
        desc: "音声はパートごとに届くので、すべてが終わるのを待たずに聴き始められます。",
      },
      download: {
        title: "ダウンロードしてシェア",
        desc: "エピソードをテキスト、音声（.wav）、動画として書き出し、ワンクリックでシェアできます。",
      },
      privacy: {
        title: "はじめからプライベート",
        desc: "APIキーも履歴書も、あなたのブラウザから直接使われ、こちら側のサーバーを経由しません。",
      },
    },
  },
  apikey: {
    title: "あなたのAPIキー",
    inputLabel: "Google AI APIキー",
    placeholder: "ここにAPIキーを貼り付け",
    remember: "このブラウザに記憶する",
    getKey: "無料でAPIキーを取得 ↗",
    note: "キーはブラウザから直接GoogleのAPIに対して使われます。「記憶する」を有効にすると、このデバイスにのみ保存されます。",
  },
  cv: {
    title: "あなたの履歴書",
    dropDrag: "履歴書をここにドラッグ、またはクリックして選択",
    dropFormats: "PDF、DOCX、TXT、または画像(PNG/JPG/WebP)",
    sizeReplace: "{size} KB · クリックで差し替え",
    processing: "ファイルを処理中...",
    ocring: "Gemini(OCR)でドキュメントからテキストを抽出中...",
    ocrButton: "🔍 AIでテキストを抽出(OCR)",
    textLabel: "履歴書のテキスト",
    clearFile: "ファイルを削除",
    placeholderFile:
      "履歴書から抽出したテキストがここに表示されます。エピソードを生成する前に編集できます...",
    placeholderManual: "...または履歴書のテキストをここに直接貼り付け",
    errors: {
      reupload:
        "AIでテキストを抽出するには、ファイルをもう一度アップロードしてください。",
      ocrNeedsKey:
        "OCR(AI)でテキストを抽出するには、APIキー(ステップ1)を入力してください。",
      ocrFailed: "AIでのテキスト抽出に失敗しました: {reason}",
      ocrEmpty: "モデルがテキストを返しませんでした",
      pdfScanned:
        "PDFからテキストを抽出できませんでした(スキャン画像のようです)。APIキーを入力して「AIでテキストを抽出(OCR)」を押すか、テキストを手動で貼り付けてください。",
      imageNeedsOcr:
        "画像から履歴書を読み取るにはAIによるOCRを使います。APIキー(ステップ1)を入力して「AIでテキストを抽出(OCR)」を押してください。",
      docxEmpty:
        "DOCXからテキストを抽出できませんでした。履歴書を手動で貼り付けてください。",
      txtEmpty:
        "TXTファイルが空です。内容を追加するか、履歴書を手動で貼り付けてください。",
      unsupported:
        "サポートされていないファイル形式です。PDF、DOCX、TXT、または画像(PNG/JPG/WebP)を使ってください。",
      processFailed: "ファイルの処理エラー: {reason}",
    },
  },
  episode: {
    title: "あなたのエピソード",
    generate: "🎭 エピソードを生成",
    regenerate: "🔁 新しいエピソードを生成",
    generating: "エピソードを生成中...",
    missingKey: "APIキーがありません(ステップ1)。",
    missingCv: "履歴書のテキストがありません(ステップ2)。",
    waitVideo: "動画の書き出しが終わるまでお待ちください。",
    scriptWriting: "執筆中...",
    scriptReady: "台本",
    copy: "📋 コピー",
    copied: "✓ コピーしました",
    share: "📣 シェア",
    linkCopied: "✓ リンクをコピーしました",
    audioFile: "🎵 音声(.wav)",
    scriptFile: "📄 台本(.txt)",
    video: "🎬 動画",
    cancelVideo: "✕ 動画をキャンセル",
    newEpisode: "✨ 新しいエピソード",
    shareText: "私の履歴書のお笑いエピソードを聴いてみて 🎙️😂",
    progress: {
      video:
        "動画をリアルタイムで録画中(エピソードの長さぶんだけかかります)...",
      writing: "台本を執筆中...",
      writingWith: "{model}で台本を執筆中...",
      recording: "エピソードを録音中...",
      recordingPart: "パート{current}/{total}を録音中...",
      partsReady: "{label}({done}/{total}パート完了)",
      preparingAudio: "音声を準備中...",
    },
    errors: {
      scriptFailed: "エピソードの生成エラー: {reason}",
      scriptEmpty: "モデルがエピソードの台本を返しませんでした",
      audioFailed:
        "音声の生成エラー: {reason} 生成済みの分は保持されます。中断したところから再開できます。",
      noAudio: "レスポンスに音声が含まれていません",
      videoFailed: "動画を書き出せませんでした: {reason}",
      copyFailed: "クリップボードにコピーできませんでした。",
      resume: "🔁 音声を再開(パート{part}から)",
      retryAudio: "🔁 音声を再試行",
    },
  },
  api: {
    unavailable:
      "現在Geminiのモデルが混み合っています(エラー503)。たいていは一時的なものです。数分待ってから再試行してください。",
    quota:
      "APIキーのクォータ上限に達しました(エラー429)。1分待ってから再試行するか、Google AI Studioでプランを確認してください。",
    invalidKey:
      "APIキーが無効か、権限がありません。Google AI Studioで確認してください。",
    unknown: "不明なエラー",
  },
  player: {
    complete: "エピソード完成",
    playBlocked: "再生を押して続行(パート{current}/{total})",
    waiting: "エピソードの次のパートを待機中...",
    part: "パート{current}/{total} · 残りを生成しながら再生できます",
    preparing: "音声の最初のパートを準備中...",
    empty: "エピソードの音声がここに表示されます",
    cover: "エピソードのカバー",
    audioLabel: "エピソードの音声",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ クスッと笑えた?",
    text: "GitHub Sponsorsでプロジェクトを応援して、新しいエピソードを届け続けよう。",
    button: "GitHub Sponsorsで支援する",
  },
  prompt: {
    script:
      'The CV Comedy Podcastへようこそ。すべての履歴書が新しいエピソードになります。あなたはポッドキャスト向けの知的でユーモラスなコンテンツ制作を専門とするプロのお笑いコンビです。ある履歴書を面白く皮肉たっぷりに批評する4〜6分のエピソードの台本を作ってください(履歴書はこのエピソードのゲストです)。\n\nマルチスピーカーTTSに必要なフォーマット:\nAlex: [1人目のホストのセリフ]\nSam: [2人目のホストのセリフ]\n\nホストのキャラクター:\n- Alex: 分析的で皮肉屋、的確で技術的な指摘をする\n- Sam: 気まぐれで陽気、面白いコメントやくだけた観察をする\n\n知的なユーモアで批評すべき要素:\n- ありがちなクリシェ:「かなりの完璧主義者です」「チームでうまく働けます」\n- 時系列や論理の矛盾\n- 誇張されたスキル:「なんでもエキスパート」\n- 平凡な仕事の大げさな説明\n- 曖昧なキャリア目標:「専門的に成長したい」\n- 関係のない、またはありきたりな趣味\n- 誤字や文法ミス\n\nトーン: 皮肉たっぷりだが洗練された、深夜のコメディ番組のように。知的なユーモアを保ち、残酷になりすぎないこと。\n\n重要:\n- 各セリフには必ず "Alex:" と "Sam:" を使うこと\n- "[...]" で自然な間を入れること\n- 適切なところに "[énfasis]" で強調を加えること\n- 会話が自然に流れるようにすること\n- 最長で3分半。絶対に超えないこと\n\n時間的コンテキスト: 今日は{date}です。履歴書の日付はこの実際の日付を基準に評価してください。あなたの知識より新しい、または後の日付は時系列の矛盾ではありません。\n\nこの履歴書を分析し、かなりかなり辛口な(文字通り容赦のないロースト)エピソードの台本を日本語で作ってください:',
    vibes:
      "おまけの素材: 履歴書の元のドキュメントを添付します。コメディアンの目で眺めてください。写真、デザイン、フォント、色、全体の「雰囲気」。何か視覚的なものが良いネタになりそうなら使ってください(うまく置いた1〜2回の言及)。ただし文字どおりに描写したり、エピソードの中心に据えたりしないでください。",
    ttsStyle:
      'CV批評ポッドキャスト形式のスタンダップコメディのエピソードの音声を日本語で、深夜番組のトーンで生成してください:皮肉たっぷりだが洗練された、知的なユーモア、残酷になりすぎないこと。各セリフには必ず司会者の名前を使い、適切なところに "[...]" で自然な間、"[énfasis]" で強調を入れてください。深夜のお笑い番組のように、会話が自然に流れるようにしてください。',
    ocr: "この履歴書(レジュメ)のすべてのテキストを抽出して書き起こしてください。コメントやマークダウン形式は付けず、ドキュメントのプレーンテキストのみを返してください。構造(セクション、日付、リスト)は改行で保ってください。",
  },
  footer: {
    disclaimer:
      "⚠️ Vibe Coding(AI)で開発されたアプリです。Google AIのAPIをブラウザから直接使用します。",
    repo: "GitHubのリポジトリ",
  },
} as const;

export default ja;
