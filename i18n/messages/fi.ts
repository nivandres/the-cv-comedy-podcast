// Suomi (Finnish)
const fi = {
  meta: {
    title:
      "The CV Comedy Podcast - Muuta jokainen CV humoristiseksi jaksoksi Geminin avulla",
    description:
      "Lataa CV:si ja luo humoristinen The CV Comedy Podcast -jakso Google Gemini 3.5 Flashilla ja multi-speaker-TTS:llä",
    ogTitle: "The CV Comedy Podcast - Muuta CV:t humoristisiksi jaksoiksi",
    ogDescription:
      "Muuta CV:si humoristiseksi jaksoksi Google Gemini 3.5 Flashilla ja Multi-Speaker-TTS:llä.",
    keywords: "podcast, CV, komedia, Google Gemini, TTS, tekoäly, huumori",
  },
  header: {
    tagline: "CV:si on vieras. Gemini kirjoittaa roastin ja antaa sille äänet.",
    badges: {
      tts: "✨ Multi-Speaker TTS",
      gemini: "🤖 Gemini 3.5 Flash",
      humor: "😄 Älykästä huumoria",
    },
  },
  theme: {
    toDark: "Vaihda tummaan teemaan",
    toLight: "Vaihda vaaleaan teemaan",
    dark: "Tumma teema",
    light: "Vaalea teema",
  },
  language: { label: "Kieli" },
  a11y: { step: "Vaihe {number}: {title}" },
  apikey: {
    title: "API-avaimesi",
    inputLabel: "Google AI API-avain",
    placeholder: "Liitä API-avaimesi tähän",
    remember: "Muista tässä selaimessa",
    getKey: "Hanki ilmainen API-avain ↗",
    note: "Avainta käytetään suoraan selaimestasi Googlen API:a vasten. Jos valitset «Muista», se tallennetaan vain tälle laitteelle.",
  },
  cv: {
    title: "CV:si",
    dropDrag: "Raahaa CV:si tähän tai valitse napsauttamalla",
    dropFormats: "PDF, DOCX, TXT tai kuva (PNG/JPG/WebP)",
    sizeReplace: "{size} KB · napsauta korvataksesi",
    processing: "Käsitellään tiedostoa...",
    ocring: "Puretaan tekstiä asiakirjasta Geminillä (OCR)...",
    ocrButton: "🔍 Pura teksti tekoälyllä (OCR)",
    textLabel: "CV:n teksti",
    clearFile: "Poista tiedosto",
    placeholderFile:
      "Tähän ilmestyy CV:stäsi purettu teksti. Voit muokata sitä ennen jakson luomista...",
    placeholderManual: "...tai liitä CV:si teksti suoraan tähän",
    errors: {
      reupload: "Lataa tiedosto uudelleen purkaaksesi tekstin tekoälyllä.",
      ocrNeedsKey:
        "Syötä API-avaimesi (vaihe 1) purkaaksesi tekstin OCR:llä (tekoäly).",
      ocrFailed: "Tekstin purkaminen tekoälyllä epäonnistui: {reason}",
      ocrEmpty: "Malli ei palauttanut tekstiä",
      pdfScanned:
        "PDF:stä ei voitu purkaa tekstiä (se vaikuttaa skannatulta). Syötä API-avaimesi ja paina «Pura teksti tekoälyllä (OCR)», tai liitä teksti manuaalisesti.",
      imageNeedsOcr:
        "CV:n lukemiseen kuvasta käytetään tekoäly-OCR:ää. Syötä API-avaimesi (vaihe 1) ja paina «Pura teksti tekoälyllä (OCR)».",
      docxEmpty: "DOCX:stä ei voitu purkaa tekstiä. Liitä CV manuaalisesti.",
      txtEmpty:
        "TXT-tiedosto on tyhjä. Lisää sisältöä tai liitä CV manuaalisesti.",
      unsupported:
        "Tiedostotyyppiä ei tueta. Käytä PDF-, DOCX-, TXT- tai kuvatiedostoa (PNG/JPG/WebP).",
      processFailed: "Virhe tiedoston käsittelyssä: {reason}",
    },
  },
  episode: {
    title: "Jaksosi",
    generate: "🎭 Luo jakso",
    regenerate: "🔁 Luo uusi jakso",
    generating: "Luodaan jaksoa...",
    missingKey: "API-avaimesi puuttuu (vaihe 1).",
    missingCv: "CV:si teksti puuttuu (vaihe 2).",
    waitVideo: "Odota, että videon vienti valmistuu.",
    scriptWriting: "Kirjoitetaan...",
    scriptReady: "Käsikirjoitus",
    copy: "📋 Kopioi",
    copied: "✓ Kopioitu",
    share: "📣 Jaa",
    linkCopied: "✓ Linkki kopioitu",
    audioFile: "🎵 Ääni (.wav)",
    scriptFile: "📄 Käsikirjoitus (.txt)",
    video: "🎬 Video",
    cancelVideo: "✕ Peruuta video",
    newEpisode: "✨ Uusi jakso",
    shareText: "Kuuntele humoristinen jakso CV:stäni 🎙️😂",
    progress: {
      video:
        "Tallennetaan videota reaaliajassa (kestää yhtä kauan kuin jakso)...",
      writing: "Kirjoitetaan käsikirjoitusta...",
      writingWith: "Kirjoitetaan käsikirjoitusta mallilla {model}...",
      recording: "Tallennetaan jaksoa...",
      recordingPart: "Tallennetaan osaa {current}/{total}...",
      partsReady: "{label} ({done}/{total} osaa valmiina)",
      preparingAudio: "Valmistellaan ääntä...",
    },
    errors: {
      scriptFailed: "Virhe jakson luomisessa: {reason}",
      scriptEmpty: "Malli ei palauttanut jakson käsikirjoitusta",
      audioFailed:
        "Virhe äänen luomisessa: {reason} Jo luotu säilyy: voit jatkaa siitä, mihin jäätiin.",
      noAudio: "Vastaus ei sisällä ääntä",
      videoFailed: "Videota ei voitu viedä: {reason}",
      copyFailed: "Leikepöydälle kopioiminen epäonnistui.",
      resume: "🔁 Jatka ääntä (osasta {part})",
      retryAudio: "🔁 Yritä ääntä uudelleen",
    },
  },
  api: {
    unavailable:
      "Gemini-mallit ovat juuri nyt ylikuormitettuja (virhe 503). Yleensä tilapäistä: odota pari minuuttia ja yritä uudelleen.",
    quota:
      "Saavutit API-avaimesi kiintiörajan (virhe 429). Odota minuutti ja yritä uudelleen, tai tarkista tilauksesi Google AI Studiossa.",
    invalidKey:
      "API-avain ei ole kelvollinen tai siltä puuttuvat oikeudet. Tarkista se Google AI Studiossa.",
    unknown: "Tuntematon virhe",
  },
  player: {
    complete: "Jakso valmis",
    playBlocked: "Paina play jatkaaksesi (osa {current}/{total})",
    waiting: "Odotetaan jakson seuraavaa osaa...",
    part: "Osa {current}/{total} · voit kuunnella samalla kun loput luodaan",
    preparing: "Valmistellaan äänen ensimmäistä osaa...",
    empty: "Jakson ääni ilmestyy tähän",
    cover: "Jakson kansikuva",
    audioLabel: "Jakson ääni",
    cc: "💬 CC",
  },
  sponsor: {
    title: "❤️ Naurattiko?",
    text: "Tue projektia GitHub Sponsorsissa, jotta uusia jaksoja syntyy jatkossakin.",
    button: "Tue GitHub Sponsorsissa",
  },
  prompt: {
    script:
      'Tervetuloa The CV Comedy Podcastiin. Jokainen CV on uusi jakso. Olet ammattikoomikoiden kaksikko, joka on erikoistunut luomaan älykästä humoristista sisältöä podcasteihin. Luo käsikirjoitus 4-6 minuutin jaksoon, joka arvostelee CV:tä hauskasti ja sarkastisesti (CV on jakson vieras).\n\nVAADITTU MUOTO multi-speaker-TTS:lle:\nAlex: [ensimmäisen juontajan teksti]\nSam: [toisen juontajan teksti]\n\nJUONTAJIEN OMINAISUUDET:\n- Alex: Analyyttinen ja sarkastinen, tekee tarkkoja teknisiä havaintoja\n- Sam: Spontaani ja hauska, heittää hauskoja kommentteja ja rentoja havaintoja\n\nÄLYKKÄÄLLÄ HUUMORILLA ARVOSTELTAVAT ASIAT:\n- Tyypilliset kliseet: "olen todella perfektionisti", "toimin hyvin tiimissä"\n- Ajalliset tai loogiset epäjohdonmukaisuudet\n- Liioitellut taidot: "kaiken asiantuntija"\n- Pöyhkeät kuvaukset tavallisista töistä\n- Epämääräiset uratavoitteet: "haluan kehittyä ammatillisesti"\n- Merkityksettömät tai kliseiset harrastukset\n- Kirjoitus- tai kielioppivirheet\n\nSÄVY: Sarkastinen mutta hienostunut, kuin myöhäisillan komediashow. Pidä huumori älykkäänä ja vältä julmuutta.\n\nTÄRKEÄÄ:\n- Käytä TÄSMÄLLEEN muotoja "Alex:" ja "Sam:" jokaisessa vuorossa\n- Sisällytä luonnollisia taukoja merkinnällä "[...]"\n- Lisää painotusta merkinnällä "[énfasis]" siellä missä se sopii\n- Anna keskustelun soljua luonnollisesti\n- Enintään kolme ja puoli minuuttia. ENINTÄÄN\n\nAJALLINEN KONTEKSTI: Tänään on {date}. Arvioi CV:n päivämääriä suhteessa tähän todelliseen päivämäärään: äskettäinen tai tietämyksesi jälkeinen päivämäärä EI ole ajallinen epäjohdonmukaisuus.\n\nAnalysoi tämä CV ja luo jakson käsikirjoitus suomeksi, ja ole todella todella kriittinen (kirjaimellisesti armoton roast):',
    vibes:
      'LISÄMATERIAALI: liitteenä on CV:n alkuperäinen asiakirja. Katso sitä koomikon silmin: valokuva, ulkoasu, typografia, värit, yleinen "vibe". Jos jokin visuaalinen antaa aihetta hyvään vitsiin, käytä sitä (yksi tai kaksi hyvin sijoitettua mainintaa), mutta älä kuvaile sitä kirjaimellisesti äläkä tee siitä jakson keskipistettä.',
    ttsStyle:
      'Luo standup-komediajakson ääni suomeksi, kriittisen CV-podcastin muodossa. Sävy on kuin myöhäisillan komediashow: sarkastinen mutta hienostunut, älykäs huumori, vältä julmuutta. Käytä täsmälleen juontajien nimiä jokaisessa vuorossa, sisällytä luonnollisia taukoja merkinnällä "[...]" ja painotusta merkinnällä "[énfasis]" siellä missä se sopii. Anna keskustelun soljua luonnollisesti, kuin myöhäisillan komediashow.',
    ocr: "Pura ja litteroi KAIKKI tämän CV:n (ansioluettelon) teksti. Palauta ainoastaan asiakirjan pelkkä teksti, ilman kommentteja tai markdown-muotoilua. Säilytä rakenne (osiot, päivämäärät, luettelot) rivinvaihdoilla.",
  },
  footer: {
    disclaimer:
      "⚠️ Sovellus kehitetty Vibe Codingilla (tekoäly). Käyttää Google AI:n API:a suoraan selaimesta.",
    repo: "GitHub-repositorio",
  },
} as const;

export default fi;
