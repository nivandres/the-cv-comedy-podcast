// Tiếng Việt (Vietnamese)
const vi = {
  meta: {
    title:
      "The CV Comedy Podcast - Biến mọi CV thành một tập podcast hài hước với Gemini",
    description:
      "Tải CV của bạn lên và tạo một tập podcast hài hước của The CV Comedy Podcast bằng Google Gemini 3.5 Flash với TTS đa giọng nói",
    ogTitle: "The CV Comedy Podcast - Biến CV thành những tập podcast hài hước",
    ogDescription:
      "Biến CV của bạn thành một tập podcast hài hước với Google Gemini 3.5 Flash và TTS đa giọng nói.",
    keywords:
      "podcast, CV, hài kịch, Google Gemini, TTS, trí tuệ nhân tạo, hài hước",
  },
  header: {
    tagline:
      "CV của bạn là khách mời. Gemini viết màn cà khịa và lồng giọng cho nó.",
    badges: {
      tts: "✨ TTS đa giọng nói",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Hài hước thông minh",
    },
  },
  theme: {
    toDark: "Chuyển sang giao diện tối",
    toLight: "Chuyển sang giao diện sáng",
    dark: "Giao diện tối",
    light: "Giao diện sáng",
  },
  language: { label: "Ngôn ngữ" },
  a11y: { step: "Bước {number}: {title}" },
  apikey: {
    title: "API Key của bạn",
    inputLabel: "Google AI API Key",
    placeholder: "Dán API Key của bạn vào đây",
    remember: "Ghi nhớ trên trình duyệt này",
    getKey: "Lấy API Key miễn phí ↗",
    note: "Key được dùng trực tiếp từ trình duyệt của bạn để gọi API của Google. Nếu bật «Ghi nhớ», nó chỉ được lưu trên thiết bị này.",
  },
  cv: {
    title: "CV của bạn",
    dropDrag: "Kéo CV của bạn vào đây hoặc nhấp để chọn",
    dropFormats: "PDF, DOCX, TXT hoặc hình ảnh (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · nhấp để thay thế",
    processing: "Đang xử lý tệp...",
    ocring: "Đang trích xuất văn bản từ tài liệu bằng Gemini (OCR)...",
    ocrButton: "🔍 Trích xuất văn bản bằng AI (OCR)",
    textLabel: "Nội dung CV",
    clearFile: "Xóa tệp",
    placeholderFile:
      "Văn bản trích xuất từ CV của bạn sẽ hiện ở đây. Bạn có thể chỉnh sửa trước khi tạo tập podcast...",
    placeholderManual: "...hoặc dán trực tiếp nội dung CV của bạn vào đây",
    errors: {
      reupload: "Hãy tải lại tệp để trích xuất văn bản bằng AI.",
      ocrNeedsKey:
        "Nhập API Key của bạn (bước 1) để trích xuất văn bản bằng OCR (AI).",
      ocrFailed: "Không thể trích xuất văn bản bằng AI: {reason}",
      ocrEmpty: "Mô hình không trả về văn bản nào",
      pdfScanned:
        "Không thể trích xuất văn bản từ PDF (có vẻ là bản scan). Nhập API Key của bạn rồi nhấn «Trích xuất văn bản bằng AI (OCR)», hoặc dán văn bản thủ công.",
      imageNeedsOcr:
        "Để đọc CV từ hình ảnh cần dùng OCR bằng AI. Nhập API Key của bạn (bước 1) rồi nhấn «Trích xuất văn bản bằng AI (OCR)».",
      docxEmpty: "Không thể trích xuất văn bản từ DOCX. Hãy dán CV thủ công.",
      txtEmpty: "Tệp TXT trống. Hãy thêm nội dung hoặc dán CV thủ công.",
      unsupported:
        "Loại tệp không được hỗ trợ. Hãy dùng PDF, DOCX, TXT hoặc hình ảnh (PNG/JPG/WebP).",
      processFailed: "Lỗi khi xử lý tệp: {reason}",
    },
  },
  episode: {
    title: "Tập podcast của bạn",
    generate: "🎭 Tạo tập podcast",
    regenerate: "🔁 Tạo một tập podcast mới",
    generating: "Đang tạo tập podcast...",
    missingKey: "Thiếu API Key của bạn (bước 1).",
    missingCv: "Thiếu nội dung CV của bạn (bước 2).",
    waitVideo: "Hãy đợi quá trình xuất video hoàn tất.",
    scriptWriting: "Đang viết...",
    scriptReady: "Kịch bản",
    copy: "📋 Sao chép",
    copied: "✓ Đã sao chép",
    share: "📣 Chia sẻ",
    linkCopied: "✓ Đã sao chép liên kết",
    audioFile: "🎵 Âm thanh (.wav)",
    scriptFile: "📄 Kịch bản (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Hủy video",
    newEpisode: "✨ Tập mới",
    shareText: "Nghe tập podcast hài hước về CV của tôi 🎙️😂",
    progress: {
      video:
        "Đang ghi video theo thời gian thực (mất thời gian bằng độ dài tập podcast)...",
      writing: "Đang viết kịch bản...",
      writingWith: "Đang viết kịch bản bằng {model}...",
      recording: "Đang thu âm tập podcast...",
      recordingPart: "Đang thu âm phần {current} trên {total}...",
      partsReady: "{label} ({done}/{total} phần đã xong)",
      preparingAudio: "Đang chuẩn bị âm thanh...",
    },
    errors: {
      scriptFailed: "Lỗi khi tạo tập podcast: {reason}",
      scriptEmpty: "Mô hình không trả về kịch bản của tập podcast",
      audioFailed:
        "Lỗi khi tạo âm thanh: {reason} Phần đã tạo vẫn được giữ lại: bạn có thể tiếp tục từ chỗ đang dở.",
      noAudio: "Phản hồi không chứa âm thanh",
      videoFailed: "Không thể xuất video: {reason}",
      copyFailed: "Không thể sao chép vào bộ nhớ tạm.",
      resume: "🔁 Tiếp tục âm thanh (từ phần {part})",
      retryAudio: "🔁 Thử lại âm thanh",
    },
  },
  api: {
    unavailable:
      "Các mô hình Gemini đang quá tải vào lúc này (lỗi 503). Thường chỉ là tạm thời: hãy đợi vài phút rồi thử lại.",
    quota:
      "Bạn đã đạt giới hạn hạn ngạch của API Key (lỗi 429). Hãy đợi một phút rồi thử lại, hoặc kiểm tra gói của bạn trong Google AI Studio.",
    invalidKey:
      "API Key không hợp lệ hoặc không có quyền. Hãy kiểm tra lại trong Google AI Studio.",
    unknown: "Lỗi không xác định",
  },
  player: {
    complete: "Tập podcast hoàn chỉnh",
    playBlocked: "Nhấn play để tiếp tục (phần {current} trên {total})",
    waiting: "Đang chờ phần tiếp theo của tập podcast...",
    part: "Phần {current} trên {total} · bạn có thể nghe trong khi phần còn lại đang được tạo",
    preparing: "Đang chuẩn bị phần đầu tiên của âm thanh...",
    empty: "Âm thanh của tập podcast sẽ hiện ở đây",
    cover: "Ảnh bìa của tập podcast",
    audioLabel: "Âm thanh của tập podcast",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Có làm bạn bật cười không?",
    text: "Hãy ủng hộ dự án trên GitHub Sponsors để những tập podcast tiếp tục được lên sóng.",
    button: "Tài trợ trên GitHub Sponsors",
  },
  prompt: {
    script:
      'Chào mừng đến với The CV Comedy Podcast. Mỗi CV là một tập podcast mới. Bạn là một cặp đôi diễn viên hài chuyên nghiệp, chuyên tạo nội dung hài hước thông minh cho podcast. Hãy viết kịch bản cho một tập dài 4-6 phút, châm biếm một CV theo cách vui nhộn và mỉa mai (CV chính là khách mời của tập này).\n\nĐỊNH DẠNG BẮT BUỘC cho TTS đa giọng nói:\nAlex: [lời của host thứ nhất]\nSam: [lời của host thứ hai]\n\nĐẶC ĐIỂM CỦA CÁC HOST:\n- Alex: Phân tích và mỉa mai, đưa ra những nhận xét kỹ thuật sắc bén\n- Sam: Ngẫu hứng và vui tính, đưa ra những bình luận hài hước và nhận xét tự nhiên\n\nNHỮNG ĐIỂM CẦN CÀ KHỊA BẰNG SỰ HÀI HƯỚC THÔNG MINH:\n- Những câu sáo rỗng điển hình: "tôi rất cầu toàn", "tôi làm việc nhóm rất tốt"\n- Những điểm mâu thuẫn về thời gian hoặc logic\n- Kỹ năng thổi phồng: "chuyên gia mọi lĩnh vực"\n- Những mô tả hoa mỹ cho các công việc hết sức bình thường\n- Mục tiêu nghề nghiệp mơ hồ: "tôi muốn phát triển bản thân"\n- Sở thích không liên quan hoặc quá sáo rỗng\n- Lỗi chính tả hoặc ngữ pháp\n\nGIỌNG ĐIỆU: Mỉa mai nhưng tinh tế, như một talkshow hài đêm khuya. Giữ sự hài hước thông minh và tránh cay nghiệt.\n\nQUAN TRỌNG:\n- Dùng CHÍNH XÁC "Alex:" và "Sam:" cho mỗi lượt thoại\n- Thêm những khoảng ngừng tự nhiên bằng "[...]"\n- Thêm nhấn mạnh bằng "[énfasis]" ở những chỗ phù hợp\n- Làm cho cuộc trò chuyện diễn ra thật tự nhiên\n- Tối đa 3 phút rưỡi. TỐI ĐA\n\nBỐI CẢNH THỜI GIAN: Hôm nay là {date}. Hãy đánh giá các mốc thời gian trong CV dựa trên ngày thực này: một ngày gần đây hoặc sau thời điểm kiến thức của bạn KHÔNG phải là điểm mâu thuẫn về thời gian.\n\nHãy phân tích CV này và viết kịch bản của tập podcast bằng tiếng Việt, và cực kỳ cực kỳ gay gắt (đúng nghĩa là một màn cà khịa không thương tiếc):',
    vibes:
      'TÀI LIỆU BỔ SUNG: đính kèm là tài liệu gốc của CV. Hãy nhìn nó bằng con mắt của một diễn viên hài: tấm ảnh, thiết kế, kiểu chữ, màu sắc, cái "vibe" tổng thể. Nếu có chi tiết hình ảnh nào đáng để làm một câu đùa hay, hãy dùng nó (một hoặc hai lần nhắc đến thật đúng chỗ), nhưng đừng mô tả nó theo nghĩa đen hay biến nó thành trọng tâm của tập podcast.',
    ttsStyle:
      'Hãy tạo âm thanh cho một tập hài standup bằng tiếng Việt, theo định dạng podcast cà khịa CV, với giọng điệu của một talkshow đêm khuya: mỉa mai nhưng tinh tế, hài hước thông minh, tránh cay nghiệt. Hãy dùng chính xác tên của các người dẫn chương trình cho mỗi lượt thoại, thêm những khoảng ngừng tự nhiên bằng "[...]" và nhấn mạnh bằng "[énfasis]" ở những chỗ phù hợp. Làm cho cuộc trò chuyện diễn ra thật tự nhiên, như một show hài đêm khuya.',
    ocr: "Hãy trích xuất và ghi lại TOÀN BỘ văn bản của CV (sơ yếu lý lịch) này. Chỉ trả về văn bản thuần của tài liệu, không kèm bình luận hay định dạng markdown. Giữ nguyên cấu trúc (các mục, ngày tháng, danh sách) bằng các dấu xuống dòng.",
  },
  footer: {
    disclaimer:
      "⚠️ Ứng dụng được phát triển bằng Vibe Coding (AI) Dùng API của Google AI trực tiếp từ trình duyệt.",
    repo: "Kho mã nguồn trên GitHub",
  },
} as const;

export default vi;
