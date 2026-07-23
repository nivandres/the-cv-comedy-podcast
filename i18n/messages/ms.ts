// Bahasa Melayu (Malay)
const ms = {
  meta: {
    title:
      "The CV Comedy Podcast - Tukar setiap CV jadi episod komedi dengan Gemini",
    description:
      "Muat naik CV anda dan jana episod komedi The CV Comedy Podcast menggunakan Google Gemini 3.5 Flash dengan TTS berbilang penutur",
    ogTitle: "The CV Comedy Podcast - Tukar CV jadi episod komedi",
    ogDescription:
      "Tukar CV anda jadi episod komedi dengan Google Gemini 3.5 Flash dan TTS Berbilang Penutur.",
    keywords:
      "podcast, CV, komedi, Google Gemini, TTS, kecerdasan buatan, jenaka",
  },
  header: {
    tagline:
      "CV anda ialah tetamunya. Gemini menulis roast dan menghidupkan suaranya.",
    badges: {
      tts: "✨ TTS Berbilang Penutur",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Jenaka bijak",
    },
  },
  theme: {
    toDark: "Tukar ke tema gelap",
    toLight: "Tukar ke tema cerah",
    dark: "Tema gelap",
    light: "Tema cerah",
  },
  language: { label: "Bahasa" },
  a11y: { step: "Langkah {number}: {title}" },
  apikey: {
    title: "API Key anda",
    inputLabel: "Google AI API Key",
    placeholder: "Tampal API Key anda di sini",
    remember: "Ingat dalam pelayar ini",
    getKey: "Dapatkan API Key percuma ↗",
    note: "Key digunakan terus dari pelayar anda ke API Google. Jika anda hidupkan «Ingat», ia disimpan hanya pada peranti ini.",
  },
  cv: {
    title: "CV anda",
    dropDrag: "Seret CV anda ke sini atau klik untuk pilih",
    dropFormats: "PDF, DOCX, TXT atau imej (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klik untuk gantikan",
    processing: "Memproses fail...",
    ocring: "Mengekstrak teks dokumen dengan Gemini (OCR)...",
    ocrButton: "🔍 Ekstrak teks dengan AI (OCR)",
    textLabel: "Teks CV",
    clearFile: "Buang fail",
    placeholderFile:
      "Teks yang diekstrak dari CV anda akan muncul di sini. Anda boleh menyuntingnya sebelum menjana episod...",
    placeholderManual: "...atau tampal terus teks CV anda di sini",
    errors: {
      reupload: "Muat naik semula fail untuk mengekstrak teks dengan AI.",
      ocrNeedsKey:
        "Masukkan API Key anda (langkah 1) untuk mengekstrak teks dengan OCR (AI).",
      ocrFailed: "Gagal mengekstrak teks dengan AI: {reason}",
      ocrEmpty: "Model tidak memulangkan sebarang teks",
      pdfScanned:
        "Gagal mengekstrak teks dari PDF (nampaknya diimbas). Masukkan API Key anda dan tekan «Ekstrak teks dengan AI (OCR)», atau tampal teks secara manual.",
      imageNeedsOcr:
        "Untuk membaca CV daripada imej, OCR dengan AI digunakan. Masukkan API Key anda (langkah 1) dan tekan «Ekstrak teks dengan AI (OCR)».",
      docxEmpty: "Gagal mengekstrak teks dari DOCX. Tampal CV secara manual.",
      txtEmpty:
        "Fail TXT kosong. Tambah kandungan atau tampal CV secara manual.",
      unsupported:
        "Jenis fail tidak disokong. Gunakan PDF, DOCX, TXT atau imej (PNG/JPG/WebP).",
      processFailed: "Ralat semasa memproses fail: {reason}",
    },
  },
  episode: {
    title: "Episod anda",
    generate: "🎭 Jana episod",
    regenerate: "🔁 Jana episod baharu",
    generating: "Menjana episod...",
    missingKey: "API Key anda tiada (langkah 1).",
    missingCv: "Teks CV anda tiada (langkah 2).",
    waitVideo: "Tunggu sehingga eksport video selesai.",
    scriptWriting: "Menulis...",
    scriptReady: "Skrip",
    copy: "📋 Salin",
    copied: "✓ Disalin",
    share: "📣 Kongsi",
    linkCopied: "✓ Pautan disalin",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Skrip (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Batal video",
    newEpisode: "✨ Episod baharu",
    shareText: "Dengar episod komedi CV saya 🎙️😂",
    progress: {
      video:
        "Merakam video secara masa nyata (mengambil masa selama episod berlangsung)...",
      writing: "Menulis skrip...",
      writingWith: "Menulis skrip dengan {model}...",
      recording: "Merakam episod...",
      recordingPart: "Merakam bahagian {current} daripada {total}...",
      partsReady: "{label} ({done}/{total} bahagian siap)",
      preparingAudio: "Menyediakan audio...",
    },
    errors: {
      scriptFailed: "Ralat semasa menjana episod: {reason}",
      scriptEmpty: "Model tidak memulangkan skrip episod",
      audioFailed:
        "Ralat semasa menjana audio: {reason} Yang sudah dihasilkan dikekalkan: anda boleh menyambung dari tempat ia terhenti.",
      noAudio: "Respons tidak mengandungi audio",
      videoFailed: "Gagal mengeksport video: {reason}",
      copyFailed: "Gagal menyalin ke papan keratan.",
      resume: "🔁 Sambung audio (dari bahagian {part})",
      retryAudio: "🔁 Cuba semula audio",
    },
  },
  api: {
    unavailable:
      "Model Gemini sedang sesak buat masa ini (ralat 503). Biasanya sementara sahaja: tunggu beberapa minit dan cuba lagi.",
    quota:
      "Anda telah mencapai had kuota API Key anda (ralat 429). Tunggu seminit dan cuba lagi, atau semak pelan anda di Google AI Studio.",
    invalidKey:
      "API Key tidak sah atau tiada kebenaran. Semak semula di Google AI Studio.",
    unknown: "Ralat tidak diketahui",
  },
  player: {
    complete: "Episod lengkap",
    playBlocked:
      "Tekan main untuk teruskan (bahagian {current} daripada {total})",
    waiting: "Menunggu bahagian episod seterusnya...",
    part: "Bahagian {current} daripada {total} · anda boleh mendengar sementara selebihnya dijana",
    preparing: "Menyediakan bahagian pertama audio...",
    empty: "Audio episod akan muncul di sini",
    cover: "Kulit episod",
    audioLabel: "Audio episod",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Buat anda ketawa?",
    text: "Sokong projek ini di GitHub Sponsors supaya episod terus mengudara.",
    button: "Jadi penaja di GitHub Sponsors",
  },
  prompt: {
    script:
      'Selamat datang ke The CV Comedy Podcast. Setiap CV ialah episod baharu. Anda ialah duo pelawak profesional yang pakar mencipta kandungan jenaka bijak untuk podcast. Hasilkan skrip untuk episod selama 4-6 minit yang mengkritik sebuah CV secara lucu dan sarkastik (CV itu ialah tetamu episod ini).\n\nFORMAT DIPERLUKAN untuk multi-speaker TTS:\nAlex: [teks host pertama]\nSam: [teks host kedua]\n\nCIRI-CIRI HOST:\n- Alex: Analitikal dan sarkastik, membuat pemerhatian teknikal yang tepat\n- Sam: Spontan dan kelakar, membuat komen lucu dan pemerhatian santai\n\nPERKARA UNTUK DIKRITIK DENGAN JENAKA BIJAK:\n- Klise biasa: "saya seorang yang sangat perfeksionis", "saya bekerja dengan baik dalam pasukan"\n- Ketidakselarasan masa atau logik\n- Kemahiran yang dibesar-besarkan: "pakar dalam segala-galanya"\n- Deskripsi bombastik untuk kerja-kerja biasa\n- Matlamat kerjaya yang kabur: "saya ingin berkembang secara profesional"\n- Hobi yang tidak relevan atau klise\n- Kesalahan ejaan atau tatabahasa\n\nNADA: Sarkastik tetapi canggih, seperti rancangan komedi late-night. Kekalkan jenaka yang bijak dan elakkan bersikap kejam.\n\nPENTING:\n- Gunakan TEPAT "Alex:" dan "Sam:" untuk setiap dialog\n- Sertakan jeda semula jadi dengan "[...]"\n- Tambah penegasan dengan "[énfasis]" di tempat yang sesuai\n- Buat perbualan mengalir secara semula jadi\n- Maksimum 3 minit setengah. MAKSIMUM\n\nKONTEKS MASA: Hari ini ialah {date}. Nilai tarikh-tarikh dalam CV berdasarkan tarikh sebenar ini: tarikh yang terkini atau selepas pengetahuan anda BUKAN satu ketidakselarasan masa.\n\nAnalisis CV ini dan hasilkan skrip episod dalam bahasa Melayu, dan cukup cukup kritikal (secara literal satu roast tanpa belas kasihan):',
    vibes:
      'BAHAN TAMBAHAN: dokumen asal CV disertakan sebagai lampiran. Lihatnya dengan mata seorang pelawak: gambar, reka bentuk, tipografi, warna, "vibe" keseluruhannya. Jika ada elemen visual yang boleh menjadi jenaka yang bagus, gunakannya (satu atau dua sebutan yang diletakkan dengan baik), tetapi jangan menerangkannya secara literal atau menjadikannya tumpuan utama episod.',
    ttsStyle:
      'Jana audio bagi sebuah episod standup comedy dalam bahasa Melayu, dalam format podcast kritikan CV, dengan nada rancangan late-night: sarkastik tetapi canggih, jenaka bijak, elakkan bersikap kejam. Gunakan tepat nama para pengacara untuk setiap dialog, sertakan jeda semula jadi dengan "[...]" dan penegasan dengan "[énfasis]" di tempat yang sesuai. Buat perbualan mengalir secara semula jadi, seperti rancangan komedi waktu malam.',
    ocr: "Ekstrak dan transkripsikan SEMUA teks daripada CV (resume) ini. Pulangkan hanya teks biasa dokumen tersebut, tanpa komen atau format markdown. Kekalkan struktur (bahagian, tarikh, senarai) dengan pemisah baris.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikasi dibangunkan dengan Vibe Coding (AI). Menggunakan API Google AI terus dari pelayar.",
    repo: "Repositori di GitHub",
  },
} as const;

export default ms;
