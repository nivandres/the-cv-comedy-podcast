// English
const en = {
  meta: {
    title:
      "The CV Comedy Podcast - Turn any resume into a comedy episode with Gemini",
    description:
      "Upload your resume and generate a comedy episode of The CV Comedy Podcast using Google Gemini 3.5 Flash with multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Turn resumes into comedy episodes",
    ogDescription:
      "Turn your resume into a comedy episode with Google Gemini 3.5 Flash and Multi-Speaker TTS.",
    keywords:
      "podcast, resume, CV, comedy, Google Gemini, TTS, artificial intelligence, humor",
  },
  header: {
    tagline: "Your resume is the guest. Gemini writes the roast and voices it.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Smart humor",
    },
  },
  theme: {
    toDark: "Switch to dark theme",
    toLight: "Switch to light theme",
    dark: "Dark theme",
    light: "Light theme",
  },
  language: { label: "Language" },
  a11y: {
    step: "Step {number}: {title}",
    skip: "Skip to content",
  },
  features: {
    toggle: "What is this?",
    heading: "Turn your resume into a comedy episode",
    intro:
      "The CV Comedy Podcast takes your resume and generates an episode where two hosts riff on it with smart humor, script and voices included. Everything is processed in your browser with Google Gemini.",
    items: {
      formats: {
        title: "Any format",
        desc: "Upload your resume as PDF, DOCX, TXT or an image; if needed, the text is extracted with AI OCR.",
      },
      script: {
        title: "AI-written script",
        desc: "Gemini writes a sharp, funny script with the tone of a late-night show.",
      },
      voices: {
        title: "Multiple voices",
        desc: "The episode is voiced by two hosts with distinct voices (multi-speaker TTS).",
      },
      streaming: {
        title: "Listen as it's generated",
        desc: "The audio arrives in parts: you start listening without waiting for the whole thing to finish.",
      },
      download: {
        title: "Download and share",
        desc: "Export the episode as text, audio (.wav) or video, and share it in one click.",
      },
      privacy: {
        title: "Private by design",
        desc: "Your API Key and your resume are used directly from your browser, without going through a server of our own.",
      },
    },
  },
  apikey: {
    title: "Your API Key",
    inputLabel: "Google AI API Key",
    placeholder: "Paste your API Key here",
    remember: "Remember on this browser",
    getKey: "Get a free API Key ↗",
    note: "Your key is used directly from your browser against the Google API. If you enable “Remember”, it is stored only on this device.",
  },
  cv: {
    title: "Your resume",
    dropDrag: "Drag your resume here or click to select",
    dropFormats: "PDF, DOCX, TXT or image (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · click to replace it",
    processing: "Processing file...",
    ocring: "Extracting text from the document with Gemini (OCR)...",
    ocrButton: "🔍 Extract text with AI (OCR)",
    textLabel: "Resume text",
    clearFile: "Remove file",
    placeholderFile:
      "The text extracted from your resume will appear here. You can edit it before generating the episode...",
    placeholderManual: "...or paste your resume text here directly",
    errors: {
      reupload: "Upload the file again to extract the text with AI.",
      ocrNeedsKey: "Enter your API Key (step 1) to extract text with OCR (AI).",
      ocrFailed: "Could not extract the text with AI: {reason}",
      ocrEmpty: "The model returned no text",
      pdfScanned:
        "Could not extract text from the PDF (it looks scanned). Enter your API Key and press “Extract text with AI (OCR)”, or paste the text manually.",
      imageNeedsOcr:
        "Reading a resume from an image uses AI OCR. Enter your API Key (step 1) and press “Extract text with AI (OCR)”.",
      docxEmpty:
        "Could not extract text from the DOCX. Paste your resume manually.",
      txtEmpty:
        "The TXT file is empty. Add content or paste your resume manually.",
      unsupported:
        "Unsupported file type. Use PDF, DOCX, TXT or an image (PNG/JPG/WebP).",
      processFailed: "Error processing file: {reason}",
    },
  },
  episode: {
    title: "Your episode",
    generate: "🎭 Generate episode",
    regenerate: "🔁 Generate a new episode",
    generating: "Generating episode...",
    missingKey: "Your API Key is missing (step 1).",
    missingCv: "Your resume text is missing (step 2).",
    waitVideo: "Wait for the video export to finish.",
    scriptWriting: "Writing...",
    scriptReady: "Script",
    copy: "📋 Copy",
    copied: "✓ Copied",
    share: "📣 Share",
    linkCopied: "✓ Link copied",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Script (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Cancel video",
    newEpisode: "✨ New episode",
    shareText: "Listen to the comedy episode of my resume 🎙️😂",
    progress: {
      video:
        "Recording the video in real time (takes as long as the episode)...",
      writing: "Writing the script...",
      writingWith: "Writing the script with {model}...",
      recording: "Recording the episode...",
      recordingPart: "Recording part {current} of {total}...",
      partsReady: "{label} ({done}/{total} parts ready)",
      preparingAudio: "Preparing the audio...",
    },
    errors: {
      scriptFailed: "Error generating the episode: {reason}",
      scriptEmpty: "The model did not return the episode script",
      audioFailed:
        "Error generating the audio: {reason} Everything generated so far is kept: you can resume from where it stopped.",
      noAudio: "The response contains no audio",
      videoFailed: "Could not export the video: {reason}",
      copyFailed: "Could not copy to the clipboard.",
      resume: "🔁 Resume audio (from part {part})",
      retryAudio: "🔁 Retry audio",
    },
  },
  api: {
    unavailable:
      "Gemini models are currently overloaded (error 503). It is usually temporary: wait a couple of minutes and try again.",
    quota:
      "You reached your API Key quota limit (error 429). Wait a minute and retry, or check your plan in Google AI Studio.",
    invalidKey:
      "The API Key is not valid or lacks permissions. Check it in Google AI Studio.",
    unknown: "Unknown error",
  },
  player: {
    complete: "Full episode",
    playBlocked: "Press play to continue (part {current} of {total})",
    waiting: "Waiting for the next part of the episode...",
    part: "Part {current} of {total} · you can listen while the rest is generated",
    preparing: "Preparing the first part of the audio...",
    empty: "The episode audio will appear here",
    cover: "Episode cover",
    audioLabel: "Episode audio",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Did it make you laugh?",
    text: "Support the project on GitHub Sponsors so new episodes keep going on air.",
    button: "Sponsor on GitHub Sponsors",
  },
  prompt: {
    script:
      'Welcome to The CV Comedy Podcast. Every resume is a new episode. You are a duo of professional comedians specialized in creating smart comedy content for podcasts. Write the script for a 4-6 minute episode that roasts a resume in a funny, sarcastic way (the resume is the guest of the episode).\n\nREQUIRED FORMAT for multi-speaker TTS:\nAlex: [first host\'s lines]\nSam: [second host\'s lines]\n\nHOST PERSONALITIES:\n- Alex: Analytical and sarcastic, makes precise technical observations\n- Sam: Spontaneous and funny, makes witty comments and casual observations\n\nTHINGS TO ROAST WITH SMART HUMOR:\n- Typical clichés: "I am a perfectionist", "I am a team player"\n- Timeline or logical inconsistencies\n- Exaggerated skills: "expert in everything"\n- Pompous descriptions of basic jobs\n- Vague career goals: "looking to grow professionally"\n- Irrelevant or cliché hobbies\n- Spelling or grammar mistakes\n\nTONE: Sarcastic but sophisticated, like a late-night comedy show. Keep the humor smart and avoid being cruel.\n\nIMPORTANT:\n- Use EXACTLY "Alex:" and "Sam:" for each line\n- Include natural pauses with "[...]"\n- Add emphasis with "[énfasis]" where appropriate\n- Make the conversation flow naturally\n- Maximum 3 and a half minutes. MAXIMUM\n\nTIME CONTEXT: Today is {date}. Evaluate the resume dates against this real date: a recent date or one later than your knowledge is NOT a timeline inconsistency.\n\nAnalyze this resume and write the episode script in English, and be very very critical (literally a merciless roast):',
    vibes:
      "EXTRA MATERIAL: the original resume document is attached. Look at it with a comedian's eye: the photo, the design, the typography, the colors, the overall vibe. If something visual makes for a good joke, use it (one or two well-placed mentions), but do not describe it literally or make it the center of the episode.",
    ttsStyle:
      'Generate the audio of a standup comedy episode in English, in the format of a resume-roasting podcast, with the tone of a late-night show: sarcastic but sophisticated, smart humor, avoid being cruel. Use exactly the hosts\' names for each line, include natural pauses with "[...]" and emphasis with "[énfasis]" where appropriate. Make the conversation flow naturally, like a late-night comedy show.',
    ocr: "Extract and transcribe ALL the text of this resume (CV). Return only the plain text of the document, with no comments or markdown formatting. Preserve the structure (sections, dates, lists) with line breaks.",
  },
  footer: {
    disclaimer:
      "⚠️ App built with Vibe Coding (AI). It uses the Google AI API directly from your browser.",
    repo: "GitHub repository",
  },
} as const;

export default en;
