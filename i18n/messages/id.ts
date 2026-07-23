// Bahasa Indonesia (Indonesian)
const id = {
  meta: {
    title:
      "The CV Comedy Podcast - Ubah setiap CV jadi episode kocak dengan Gemini",
    description:
      "Unggah CV-mu dan buat episode kocak The CV Comedy Podcast pakai Google Gemini 3.5 Flash dengan multi-speaker TTS",
    ogTitle: "The CV Comedy Podcast - Ubah CV jadi episode kocak",
    ogDescription:
      "Ubah CV-mu jadi episode kocak dengan Google Gemini 3.5 Flash dan Multi-Speaker TTS.",
    keywords:
      "podcast, CV, komedi, Google Gemini, TTS, kecerdasan buatan, humor",
  },
  header: {
    tagline:
      "CV-mu adalah bintang tamunya. Gemini menulis roast-nya dan mengisi suaranya.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Humor cerdas",
    },
  },
  theme: {
    toDark: "Beralih ke tema gelap",
    toLight: "Beralih ke tema terang",
    dark: "Tema gelap",
    light: "Tema terang",
  },
  language: { label: "Bahasa" },
  a11y: { step: "Langkah {number}: {title}" },
  apikey: {
    title: "API Key kamu",
    inputLabel: "Google AI API Key",
    placeholder: "Tempel API Key kamu di sini",
    remember: "Ingat di browser ini",
    getKey: "Dapatkan API Key gratis ↗",
    note: "Key dipakai langsung dari browser kamu ke API Google. Kalau kamu aktifkan «Ingat», key hanya disimpan di perangkat ini.",
  },
  cv: {
    title: "CV kamu",
    dropDrag: "Seret CV kamu ke sini atau klik untuk memilih",
    dropFormats: "PDF, DOCX, TXT atau gambar (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · klik untuk mengganti",
    processing: "Memproses file...",
    ocring: "Mengekstrak teks dari dokumen dengan Gemini (OCR)...",
    ocrButton: "🔍 Ekstrak teks dengan AI (OCR)",
    textLabel: "Teks CV",
    clearFile: "Hapus file",
    placeholderFile:
      "Teks yang diekstrak dari CV kamu akan muncul di sini. Kamu bisa mengeditnya dulu sebelum membuat episode...",
    placeholderManual: "...atau tempel langsung teks CV kamu di sini",
    errors: {
      reupload: "Unggah ulang file-nya untuk mengekstrak teks dengan AI.",
      ocrNeedsKey:
        "Masukkan API Key kamu (langkah 1) untuk mengekstrak teks dengan OCR (AI).",
      ocrFailed: "Gagal mengekstrak teks dengan AI: {reason}",
      ocrEmpty: "Model tidak mengembalikan teks apa pun",
      pdfScanned:
        "Gagal mengekstrak teks dari PDF (sepertinya hasil pindaian). Masukkan API Key kamu lalu tekan «Ekstrak teks dengan AI (OCR)», atau tempel teksnya secara manual.",
      imageNeedsOcr:
        "Untuk membaca CV dari gambar, kami pakai OCR dengan AI. Masukkan API Key kamu (langkah 1) lalu tekan «Ekstrak teks dengan AI (OCR)».",
      docxEmpty:
        "Gagal mengekstrak teks dari DOCX. Tempel CV-nya secara manual.",
      txtEmpty:
        "File TXT-nya kosong. Tambahkan isinya atau tempel CV secara manual.",
      unsupported:
        "Tipe file tidak didukung. Gunakan PDF, DOCX, TXT atau gambar (PNG/JPG/WebP).",
      processFailed: "Gagal memproses file: {reason}",
    },
  },
  episode: {
    title: "Episode kamu",
    generate: "🎭 Buat episode",
    regenerate: "🔁 Buat episode baru",
    generating: "Membuat episode...",
    missingKey: "API Key kamu belum ada (langkah 1).",
    missingCv: "Teks CV kamu belum ada (langkah 2).",
    waitVideo: "Tunggu sampai ekspor videonya selesai.",
    scriptWriting: "Menulis...",
    scriptReady: "Naskah",
    copy: "📋 Salin",
    copied: "✓ Tersalin",
    share: "📣 Bagikan",
    linkCopied: "✓ Tautan tersalin",
    audioFile: "🎵 Audio (.wav)",
    scriptFile: "📄 Naskah (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Batalkan video",
    newEpisode: "✨ Episode baru",
    shareText: "Dengerin episode kocak dari CV-ku 🎙️😂",
    progress: {
      video: "Merekam video secara real-time (selama durasi episode)...",
      writing: "Menulis naskah...",
      writingWith: "Menulis naskah dengan {model}...",
      recording: "Merekam episode...",
      recordingPart: "Merekam bagian {current} dari {total}...",
      partsReady: "{label} ({done}/{total} bagian siap)",
      preparingAudio: "Menyiapkan audio...",
    },
    errors: {
      scriptFailed: "Gagal membuat episode: {reason}",
      scriptEmpty: "Model tidak mengembalikan naskah episode",
      audioFailed:
        "Gagal membuat audio: {reason} Bagian yang sudah jadi tetap tersimpan: kamu bisa melanjutkan dari titik terakhir.",
      noAudio: "Responsnya tidak berisi audio",
      videoFailed: "Gagal mengekspor video: {reason}",
      copyFailed: "Gagal menyalin ke papan klip.",
      resume: "🔁 Lanjutkan audio (dari bagian {part})",
      retryAudio: "🔁 Coba lagi audio",
    },
  },
  api: {
    unavailable:
      "Model Gemini lagi penuh banget saat ini (error 503). Biasanya cuma sementara: tunggu beberapa menit lalu coba lagi.",
    quota:
      "Kamu sudah mencapai batas kuota API Key kamu (error 429). Tunggu sebentar lalu coba lagi, atau cek paketmu di Google AI Studio.",
    invalidKey:
      "API Key tidak valid atau tidak punya izin. Cek lagi di Google AI Studio.",
    unknown: "Error tidak diketahui",
  },
  player: {
    complete: "Episode selesai",
    playBlocked: "Tekan play untuk lanjut (bagian {current} dari {total})",
    waiting: "Menunggu bagian episode berikutnya...",
    part: "Bagian {current} dari {total} · kamu bisa mendengarkan sambil sisanya dibuat",
    preparing: "Menyiapkan bagian pertama audio...",
    empty: "Audio episode akan muncul di sini",
    cover: "Sampul episode",
    audioLabel: "Audio episode",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Bikin kamu ketawa?",
    text: "Dukung proyek ini di GitHub Sponsors biar episode-episode baru terus mengudara.",
    button: "Jadi sponsor di GitHub Sponsors",
  },
  prompt: {
    script:
      'Selamat datang di The CV Comedy Podcast. Setiap CV adalah episode baru. Kamu adalah duo komedian profesional yang ahli membuat konten humor cerdas untuk podcast. Buat naskah untuk episode berdurasi 4-6 menit yang mengkritik sebuah CV dengan cara yang lucu dan sarkastis (CV itu adalah bintang tamu episode ini).\n\nFORMAT WAJIB untuk multi-speaker TTS:\nAlex: [teks host pertama]\nSam: [teks host kedua]\n\nKARAKTER PARA HOST:\n- Alex: Analitis dan sarkastis, melempar pengamatan teknis yang akurat\n- Sam: Spontan dan kocak, melontarkan komentar lucu dan pengamatan santai\n\nHAL-HAL YANG DIROAST DENGAN HUMOR CERDAS:\n- Klise klasik: "saya sangat perfeksionis", "saya jago kerja tim"\n- Ketidakkonsistenan waktu atau logika\n- Keterampilan yang dilebih-lebihkan: "ahli dalam segala hal"\n- Deskripsi bombastis untuk pekerjaan biasa\n- Tujuan karier yang samar: "ingin berkembang secara profesional"\n- Hobi yang nggak relevan atau klise\n- Kesalahan ejaan atau tata bahasa\n\nNADA: Sarkastis tapi berkelas, seperti acara late-night comedy. Jaga humornya tetap cerdas dan hindari bersikap kejam.\n\nPENTING:\n- Gunakan PERSIS "Alex:" dan "Sam:" untuk setiap dialog\n- Sertakan jeda alami dengan "[...]"\n- Tambahkan penekanan dengan "[énfasis]" jika perlu\n- Buat percakapan mengalir secara alami\n- Maksimal 3 setengah menit. MAKSIMAL\n\nKONTEKS WAKTU: Hari ini adalah {date}. Nilai tanggal-tanggal di CV berdasarkan tanggal nyata ini: tanggal yang baru atau melewati batas pengetahuanmu BUKAN merupakan ketidakkonsistenan waktu.\n\nAnalisis CV ini dan buat naskah episodenya dalam bahasa Indonesia, dan sangat sangat kritis (secara harfiah roast tanpa ampun):',
    vibes:
      'MATERI TAMBAHAN: dokumen asli CV-nya terlampir. Lihat dengan mata seorang komedian: fotonya, desainnya, tipografinya, warnanya, "vibe" keseluruhannya. Kalau ada elemen visual yang bisa jadi lelucon bagus, pakailah (satu atau dua sentilan yang pas), tapi jangan dideskripsikan secara harfiah atau dijadikan pusat episode.',
    ttsStyle:
      'Hasilkan audio sebuah episode standup comedy dalam bahasa Indonesia, dalam format podcast yang mengkritik CV, dengan nada late-night show: sarkastis tapi berkelas, humor cerdas, hindari bersikap kejam. Gunakan persis nama para pembawa acara untuk setiap dialog, sertakan jeda alami dengan "[...]" dan penekanan dengan "[énfasis]" jika perlu. Buat percakapan mengalir secara alami, seperti acara komedi malam.',
    ocr: "Ekstrak dan transkripsikan SELURUH teks dari CV (resume) ini. Kembalikan hanya teks polos dari dokumen, tanpa komentar maupun format markdown. Pertahankan strukturnya (bagian, tanggal, daftar) dengan pergantian baris.",
  },
  footer: {
    disclaimer:
      "⚠️ Aplikasi yang dibangun dengan Vibe Coding (AI) Memakai API Google AI langsung dari browser.",
    repo: "Repositori di GitHub",
  },
} as const;

export default id;
