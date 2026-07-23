// Türkçe (Turkish)
const tr = {
  meta: {
    title:
      "The CV Comedy Podcast - Her CV'yi Gemini ile komik bir bölüme dönüştür",
    description:
      "CV'ni yükle ve çok konuşmacılı TTS'li Google Gemini 3.5 Flash ile The CV Comedy Podcast'in komik bir bölümünü oluştur",
    ogTitle: "The CV Comedy Podcast - CV'leri komik bölümlere dönüştür",
    ogDescription:
      "CV'ni Google Gemini 3.5 Flash ve Çok Konuşmacılı TTS ile komik bir bölüme dönüştür.",
    keywords: "podcast, CV, komedi, Google Gemini, TTS, yapay zeka, mizah",
  },
  header: {
    tagline:
      "CV'in bu bölümün konuğu. Roast'u Gemini yazıyor ve seslendiriyor.",
    badges: {
      tts: "✨ Çok Konuşmacılı TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Zekice mizah",
    },
  },
  theme: {
    toDark: "Koyu temaya geç",
    toLight: "Açık temaya geç",
    dark: "Koyu tema",
    light: "Açık tema",
  },
  language: { label: "Dil" },
  a11y: { step: "Adım {number}: {title}" },
  apikey: {
    title: "API Anahtarın",
    inputLabel: "Google AI API Anahtarı",
    placeholder: "API Anahtarını buraya yapıştır",
    remember: "Bu tarayıcıda hatırla",
    getKey: "Ücretsiz bir API Anahtarı al ↗",
    note: "Anahtar doğrudan tarayıcından Google'ın API'sine gönderilir. «Hatırla»yı açarsan yalnızca bu cihazda saklanır.",
  },
  cv: {
    title: "CV'in",
    dropDrag: "CV'ini buraya sürükle ya da seçmek için tıkla",
    dropFormats: "PDF, DOCX, TXT ya da görsel (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · değiştirmek için tıkla",
    processing: "Dosya işleniyor...",
    ocring: "Gemini ile belgeden metin çıkarılıyor (OCR)...",
    ocrButton: "🔍 Yapay zekayla metin çıkar (OCR)",
    textLabel: "CV metni",
    clearFile: "Dosyayı kaldır",
    placeholderFile:
      "CV'inden çıkarılan metin burada görünecek. Bölümü oluşturmadan önce düzenleyebilirsin...",
    placeholderManual: "...ya da CV metnini doğrudan buraya yapıştır",
    errors: {
      reupload: "Metni yapay zekayla çıkarmak için dosyayı tekrar yükle.",
      ocrNeedsKey:
        "Metni OCR (yapay zeka) ile çıkarmak için API Anahtarını gir (adım 1).",
      ocrFailed: "Metin yapay zekayla çıkarılamadı: {reason}",
      ocrEmpty: "Model hiç metin döndürmedi",
      pdfScanned:
        "PDF'ten metin çıkarılamadı (taranmış görünüyor). API Anahtarını gir ve «Yapay zekayla metin çıkar (OCR)»a bas ya da metni elle yapıştır.",
      imageNeedsOcr:
        "CV'yi bir görselden okumak için yapay zeka destekli OCR kullanılır. API Anahtarını gir (adım 1) ve «Yapay zekayla metin çıkar (OCR)»a bas.",
      docxEmpty: "DOCX'ten metin çıkarılamadı. CV'yi elle yapıştır.",
      txtEmpty: "TXT dosyası boş. İçerik ekle ya da CV'yi elle yapıştır.",
      unsupported:
        "Desteklenmeyen dosya türü. PDF, DOCX, TXT ya da bir görsel (PNG/JPG/WebP) kullan.",
      processFailed: "Dosya işlenirken hata: {reason}",
    },
  },
  episode: {
    title: "Bölümün",
    generate: "🎭 Bölüm oluştur",
    regenerate: "🔁 Yeni bir bölüm oluştur",
    generating: "Bölüm oluşturuluyor...",
    missingKey: "API Anahtarın eksik (adım 1).",
    missingCv: "CV metnin eksik (adım 2).",
    waitVideo: "Videonun dışa aktarımı bitene kadar bekle.",
    scriptWriting: "Yazılıyor...",
    scriptReady: "Senaryo",
    copy: "📋 Kopyala",
    copied: "✓ Kopyalandı",
    share: "📣 Paylaş",
    linkCopied: "✓ Bağlantı kopyalandı",
    audioFile: "🎵 Ses (.wav)",
    scriptFile: "📄 Senaryo (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Videoyu iptal et",
    newEpisode: "✨ Yeni bölüm",
    shareText: "CV'imin komik bölümünü dinle 🎙️😂",
    progress: {
      video:
        "Video gerçek zamanlı kaydediliyor (bölüm ne kadar sürüyorsa o kadar sürer)...",
      writing: "Senaryo yazılıyor...",
      writingWith: "Senaryo {model} ile yazılıyor...",
      recording: "Bölüm kaydediliyor...",
      recordingPart: "Parça {current}/{total} kaydediliyor...",
      partsReady: "{label} ({done}/{total} parça hazır)",
      preparingAudio: "Ses hazırlanıyor...",
    },
    errors: {
      scriptFailed: "Bölüm oluşturulurken hata: {reason}",
      scriptEmpty: "Model bölümün senaryosunu döndürmedi",
      audioFailed:
        "Ses oluşturulurken hata: {reason} Şimdiye kadar oluşturulanlar korunur: kaldığın yerden devam edebilirsin.",
      noAudio: "Yanıt ses içermiyor",
      videoFailed: "Video dışa aktarılamadı: {reason}",
      copyFailed: "Panoya kopyalanamadı.",
      resume: "🔁 Sesi sürdür ({part}. parçadan)",
      retryAudio: "🔁 Sesi yeniden dene",
    },
  },
  api: {
    unavailable:
      "Gemini modelleri şu anda aşırı yoğun (503 hatası). Genelde geçicidir: birkaç dakika bekleyip tekrar dene.",
    quota:
      "API Anahtarının kota sınırına ulaştın (429 hatası). Bir dakika bekleyip tekrar dene ya da Google AI Studio'daki planını kontrol et.",
    invalidKey:
      "API Anahtarı geçersiz ya da yeterli izne sahip değil. Google AI Studio'dan kontrol et.",
    unknown: "Bilinmeyen hata",
  },
  player: {
    complete: "Bölüm tamamlandı",
    playBlocked: "Devam etmek için oynat'a bas (parça {current}/{total})",
    waiting: "Bölümün sonraki parçası bekleniyor...",
    part: "Parça {current}/{total} · geri kalanı oluşturulurken dinleyebilirsin",
    preparing: "Sesin ilk parçası hazırlanıyor...",
    empty: "Bölümün sesi burada görünecek",
    cover: "Bölüm kapağı",
    audioLabel: "Bölüm sesi",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Seni güldürdü mü?",
    text: "Yeni bölümlerin yayınlanmaya devam etmesi için projeye GitHub Sponsors'tan destek ol.",
    button: "GitHub Sponsors'ta destekle",
  },
  prompt: {
    script:
      'The CV Comedy Podcast\'e hoş geldin. Her CV yeni bir bölümdür. Podcast\'ler için zekice mizahi içerik üretmekte uzmanlaşmış profesyonel bir komedyen ikilisisin. Bir CV\'yi eğlenceli ve alaycı bir dille eleştiren 4-6 dakikalık bir bölümün senaryosunu yaz (CV, bölümün konuğudur).\n\nÇok konuşmacılı TTS için GEREKLİ FORMAT:\nAlex: [birinci sunucunun metni]\nSam: [ikinci sunucunun metni]\n\nSUNUCULARIN KARAKTERİ:\n- Alex: Analitik ve alaycı, isabetli teknik gözlemler yapar\n- Sam: Doğaçlama ve komik, eğlenceli yorumlar ve rahat gözlemler yapar\n\nZEKİCE MİZAHLA ELEŞTİRİLECEK UNSURLAR:\n- Tipik klişeler: "aşırı mükemmeliyetçiyimdir", "takımla uyumlu çalışırım"\n- Zamansal ya da mantıksal tutarsızlıklar\n- Abartılı yetenekler: "her şeyin uzmanı"\n- Sıradan işlerin şatafatlı tarifleri\n- Belirsiz kariyer hedefleri: "profesyonel olarak gelişmek istiyorum"\n- Alakasız ya da klişe hobiler\n- Yazım ya da dilbilgisi hataları\n\nTON: Alaycı ama sofistike, tıpkı bir gece kuşağı komedi programı gibi. Mizahı zekice tut, acımasız olmaktan kaçın.\n\nÖNEMLİ:\n- Her replikte TAM OLARAK "Alex:" ve "Sam:" kullan\n- "[...]" ile doğal duraklamalar ekle\n- Uygun yerlerde "[énfasis]" ile vurgu kat\n- Konuşmanın doğal bir şekilde akmasını sağla\n- En fazla üç buçuk dakika. EN FAZLA\n\nZAMANSAL BAĞLAM: Bugün {date}. CV\'deki tarihleri bu gerçek tarihe göre değerlendir: güncel ya da senin bilgi kesim tarihinden sonraki bir tarih zamansal tutarsızlık DEĞİLDİR.\n\nBu CV\'yi analiz et ve bölümün senaryosunu Türkçe olarak, hem de bayağı bayağı eleştirel bir dille yaz (kelimenin tam anlamıyla acımasız bir roast):',
    vibes:
      'EK MALZEME: CV\'nin orijinal belgesi ekte. Ona bir komedyen gözüyle bak: fotoğraf, tasarım, tipografi, renkler, genel "vibe". Görsel bir şey iyi bir şakaya malzeme oluyorsa kullan (yerinde bir ya da iki değini), ama onu birebir tarif etme ya da bölümün odağı haline getirme.',
    ttsStyle:
      'CV\'leri eleştiren bir podcast formatında, gece kuşağı programı tonuyla, bir stand-up komedi bölümünün sesini Türkçe olarak üret: alaycı ama sofistike, zekice mizah, acımasız olmaktan kaçın. Her replikte sunucuların adlarını tam olarak kullan; uygun yerlerde "[...]" ile doğal duraklamalar ve "[énfasis]" ile vurgu ekle. Konuşmanın, bir gece komedi programı gibi doğal bir şekilde akmasını sağla.',
    ocr: "Bu CV'nin (özgeçmiş) TÜM metnini çıkar ve deşifre et. Yalnızca belgenin düz metnini döndür; yorum ya da markdown biçimi ekleme. Yapıyı (bölümler, tarihler, listeler) satır sonlarıyla koru.",
  },
  footer: {
    disclaimer:
      "⚠️ Vibe Coding (yapay zeka) ile geliştirilen uygulama. Google AI'nin API'sini doğrudan tarayıcıdan kullanır.",
    repo: "GitHub deposu",
  },
} as const;

export default tr;
