import { useState, useRef, useEffect } from "react";
import Head from "next/head";
import type { GetStaticPaths, GetStaticProps } from "next";
import type { Part } from "@google/genai";
import {
  pcmToWav,
  base64ToUint8Array,
  arrayBufferToBase64,
  parseSampleRate,
  concatPcm,
  downloadBlob,
  exportEpisodeVideo,
  DEFAULT_TTS_SAMPLE_RATE,
} from "@/lib/audio";
import {
  SCRIPT_MODELS,
  TTS_MODELS,
  TTS_SEGMENT_MAX_CHARS,
  callWithFallback,
  classifyApiError,
  createClient,
  splitScriptIntoSegments,
} from "@/lib/gemini";
import { useLocale } from "intl-t/react";
import {
  DEFAULT_LOCALE,
  LOCALES,
  OG_LOCALES,
  SITE_URL,
  isAppLocale,
  localeLoaders,
  localeUrl,
  useTranslation,
  type AppLocale,
} from "@/i18n/translation";
import {
  Button,
  Spinner,
  StepCard,
  ProgressBar,
  Alert,
  ThemeToggle,
  LanguageSwitcher,
  type StepStatus,
} from "@/components/ui";
import { EpisodePlayer } from "@/components/EpisodePlayer";

// Los nombres son literales en los prompts de cada locale (deben coincidir
// con las etiquetas "Alex:"/"Sam:" del libreto); las personalidades viven en
// locales/*.ts dentro de prompt.script.
interface Speaker {
  name: string;
  voice: string;
}

const SPEAKERS: Speaker[] = [
  { name: "Alex", voice: "Kore" },
  { name: "Sam", voice: "Puck" },
];

const SPEAKER_NAMES = SPEAKERS.map((speaker) => speaker.name);

const MOCK_CV_SUMMARY = `NOMBRE DEL CANDIDATO: Juan Pérez\nPROFESIÓN/ÁREA: Ingeniero de Software\nEXPERIENCIA PRINCIPAL:\n- 5 años desarrollando aplicaciones web\n- Experiencia liderando equipos ágiles\nEDUCACIÓN: Licenciatura en Ingeniería Informática\nHABILIDADES DESTACADAS: JavaScript, React, Node.js, liderazgo\nOBSERVACIONES GENERALES: CV bien estructurado, pero usa muchos clichés y frases genéricas.`;

const API_KEY_STORAGE = "gemini-api-key";
const APP_URL = "https://the-cv-comedy-podcast.vercel.app/";
// Con menos texto que esto, el PDF se considera escaneado y pasa por OCR
const MIN_PDF_TEXT_CHARS = 100;

export default function TheCVComedyPodcast() {
  const t = useTranslation();
  // Locale activo (reactivo: sigue el cambio de idioma en caliente) para las
  // etiquetas SEO — así canonical/og/hreflang concuerdan con lo que se muestra.
  const { locale: rawLocale } = useLocale();
  const locale: AppLocale = isAppLocale(rawLocale) ? rawLocale : DEFAULT_LOCALE;
  // ── Paso 1: API key ────────────────────────────────────────────────
  const [apiKey, setApiKey] = useState("");
  const [rememberKey, setRememberKey] = useState(true);

  // ── Paso 2: CV ─────────────────────────────────────────────────────
  const [file, setFile] = useState<File | null>(null);
  const [cvText, setCvText] = useState("");
  const [fileError, setFileError] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);
  const [isOcrRunning, setIsOcrRunning] = useState(false);
  const [needsOcr, setNeedsOcr] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  // Documento original (PDF o imagen): para OCR y para que el modelo pueda
  // "ver" la foto, el diseño y el vibe general del CV al escribir el libreto
  const docRef = useRef<{ bytes: ArrayBuffer; mimeType: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Invalida extracciones en vuelo cuando el usuario quita o reemplaza el
  // archivo: cada carga incrementa el id y los resultados viejos se descartan
  const uploadIdRef = useRef(0);

  // ── Paso 3: libreto ────────────────────────────────────────────────
  const [podcastScript, setPodcastScript] = useState("");
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [scriptError, setScriptError] = useState("");
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const scriptRunningRef = useRef(false);

  // ── Paso 4: audio (TTS por segmentos, reanudable) ──────────────────
  const [audioSegments, setAudioSegments] = useState<Blob[]>([]);
  const [totalSegments, setTotalSegments] = useState(0);
  const [fullAudio, setFullAudio] = useState<Blob | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [audioError, setAudioError] = useState("");
  const audioRunningRef = useRef(false);
  const pcmChunksRef = useRef<Uint8Array[]>([]);
  const segTextsRef = useRef<string[]>([]);
  const [segmentTexts, setSegmentTexts] = useState<string[]>([]);
  const sampleRateRef = useRef(DEFAULT_TTS_SAMPLE_RATE);
  const ttsModelIdxRef = useRef(0);
  const [sampleRate, setSampleRate] = useState(DEFAULT_TTS_SAMPLE_RATE);

  // ── Paso 5: exportar / compartir ───────────────────────────────────
  const [isExportingVideo, setIsExportingVideo] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoError, setVideoError] = useState("");
  const videoAbortRef = useRef<AbortController | null>(null);
  const [shareState, setShareState] = useState<"idle" | "copied">("idle");

  // ── Transversal ────────────────────────────────────────────────────
  const [activityNote, setActivityNote] = useState("");
  const [debugLog, setDebugLog] = useState<string[]>([]);
  const [devMode, setDevMode] = useState(false);

  // Mensaje de error de la API en el idioma activo (categorías conocidas
  // traducidas; desconocidas muestran el mensaje crudo)
  const apiErrorMessage = (err: unknown) => {
    const kind = classifyApiError(err);
    if (kind) return String(t.api[kind]);
    return (
      (err instanceof Error ? err.message : String(err)) ||
      String(t.api.unknown)
    );
  };

  const addLog = (message: string) => {
    if (devMode) console.log(message);
    setDebugLog((prev) => [
      ...prev.slice(-99),
      `${new Date().toLocaleTimeString()}: ${message}`,
    ]);
  };

  // Carga la API key guardada y la persiste automáticamente en este navegador
  // (lectura de localStorage tras hidratar; solo corre una vez al montar)
  useEffect(() => {
    const stored = localStorage.getItem(API_KEY_STORAGE);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (stored) setApiKey(stored);
  }, []);
  useEffect(() => {
    if (rememberKey && apiKey.trim()) {
      localStorage.setItem(API_KEY_STORAGE, apiKey.trim());
    } else {
      // Vaciar el campo (o desmarcar «Recordar») elimina la key del dispositivo
      localStorage.removeItem(API_KEY_STORAGE);
    }
  }, [apiKey, rememberKey]);

  // Modo dev (?dev=1): precarga un libreto mock sin gastar API
  useEffect(() => {
    if (!window.location.search.includes("dev=1")) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDevMode(true);
    setPodcastScript(
      "[Episodio de prueba]\nAlex: Bienvenidos a The CV Comedy Podcast.\nSam: Hoy solo estamos probando el reproductor.\n[...]"
    );
    setCvText(MOCK_CV_SUMMARY);
  }, []);

  // ── Estado derivado del flujo ──────────────────────────────────────
  const hasKey = apiKey.trim().length > 0;
  const hasCv = cvText.trim().length > 0;
  const busy = isGeneratingScript || isGeneratingAudio;
  const scriptReady = podcastScript.trim().length > 0 && !isGeneratingScript;
  const audioStarted = isGeneratingAudio || audioSegments.length > 0;
  const audioReady = Boolean(fullAudio);
  const showOcrButton = needsOcr && !hasCv && !isOcrRunning;

  const stepStatus: Record<number, StepStatus> = {
    1: hasKey ? "done" : "active",
    2: hasCv ? "done" : "active",
    3: audioReady ? "done" : hasKey && hasCv ? "active" : "locked",
  };

  // Una sola barra de progreso para todo el pipeline: el libreto cuenta como
  // la primera unidad y cada parte de audio como una más. La exportación de
  // video (opcional) reutiliza la misma barra.
  const pipeline = isExportingVideo
    ? {
        show: true,
        value: videoProgress,
        indeterminate: false,
        label: String(t.episode.progress.video),
      }
    : isGeneratingScript
      ? {
          show: true,
          value: 0,
          indeterminate: true,
          label: activityNote || String(t.episode.progress.writing),
        }
      : isGeneratingAudio
        ? {
            show: true,
            value:
              totalSegments > 0
                ? ((1 + audioSegments.length) / (1 + totalSegments)) * 100
                : 10,
            indeterminate: totalSegments === 0,
            label:
              totalSegments > 0
                ? String(
                    t.episode.progress.partsReady({
                      label:
                        activityNote || String(t.episode.progress.recording),
                      done: audioSegments.length,
                      total: totalSegments,
                    })
                  )
                : String(t.episode.progress.preparingAudio),
          }
        : { show: false, value: 0, indeterminate: false, label: "" };

  // ── Paso 2: extracción de texto ────────────────────────────────────

  // OCR con Gemini: envía el documento completo (PDF o imagen, en base64)
  // para transcribir su texto. Útil para escaneados y fotos de CVs.
  const runOcr = async () => {
    const key = apiKey.trim();
    const doc = docRef.current;
    if (!doc) {
      setFileError(String(t.cv.errors.reupload));
      return;
    }
    if (!key) {
      setFileError(String(t.cv.errors.ocrNeedsKey));
      return;
    }
    setIsOcrRunning(true);
    setFileError("");
    try {
      addLog("Extrayendo texto del documento con Gemini (OCR)...");
      const ai = await createClient(key);
      const { result } = await callWithFallback(
        SCRIPT_MODELS,
        (model) =>
          ai.models.generateContent({
            model,
            contents: [
              {
                parts: [
                  { text: String(t.prompt.ocr) },
                  {
                    inlineData: {
                      mimeType: doc.mimeType,
                      data: arrayBufferToBase64(doc.bytes),
                    },
                  },
                ],
              },
            ],
          }),
        addLog
      );
      const text = (result.text || "").trim();
      if (!text) throw new Error(String(t.cv.errors.ocrEmpty));
      // Si el usuario quitó o reemplazó el archivo mientras corría el OCR,
      // este resultado ya no es vigente: descartarlo
      if (docRef.current !== doc) {
        addLog("OCR descartado: el archivo cambió durante la extracción");
        return;
      }
      setCvText(text);
      setNeedsOcr(false);
      addLog(`OCR completado: ${text.length} caracteres extraídos`);
    } catch (ocrError) {
      addLog(`Error en OCR: ${ocrError}`);
      if (docRef.current === doc) {
        setFileError(
          String(t.cv.errors.ocrFailed({ reason: apiErrorMessage(ocrError) }))
        );
      }
    } finally {
      setIsOcrRunning(false);
    }
  };

  const clearFile = () => {
    uploadIdRef.current++; // invalida extracciones en vuelo
    setFile(null);
    setCvText("");
    setFileError("");
    setNeedsOcr(false);
    docRef.current = null;
  };

  const handleFileUpload = async (uploadedFile: File) => {
    if (busy || isExtracting || isOcrRunning) return;
    const uploadId = ++uploadIdRef.current;
    addLog(`Iniciando carga de archivo: ${uploadedFile.name}`);
    setIsExtracting(true);
    setFileError("");
    setCvText("");
    setNeedsOcr(false);
    docRef.current = null;

    try {
      const fileName = uploadedFile.name.toLowerCase();
      const isPdf =
        uploadedFile.type === "application/pdf" || fileName.endsWith(".pdf");
      const isText =
        uploadedFile.type === "text/plain" || fileName.endsWith(".txt");
      const isDocx =
        uploadedFile.type ===
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        fileName.endsWith(".docx");
      const isImage =
        uploadedFile.type.startsWith("image/") ||
        /\.(png|jpe?g|webp)$/.test(fileName);

      if (!isPdf && !isImage && !isDocx && !isText) {
        throw new Error(String(t.cv.errors.unsupported));
      }
      setFile(uploadedFile);
      // Si el usuario quita o reemplaza el archivo durante la extracción,
      // los resultados de esta carga se descartan
      const stale = () => uploadIdRef.current !== uploadId;

      if (isPdf) {
        addLog("Procesando archivo PDF...");
        // Guardamos los bytes para OCR y para que el modelo pueda "ver" el
        // documento. pdfjs transfiere el buffer al worker: usamos una copia.
        const arrayBuffer = await uploadedFile.arrayBuffer();
        if (stale()) return;
        docRef.current = {
          bytes: arrayBuffer.slice(0),
          mimeType: "application/pdf",
        };
        let fullText = "";
        try {
          const pdfjsLib = await import("pdfjs-dist");
          pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
            "pdfjs-dist/build/pdf.worker.min.mjs",
            import.meta.url
          ).toString();
          const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
          const pdf = await loadingTask.promise;
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items
              .map((item) => ("str" in item ? item.str : ""))
              .join(" ");
            fullText += pageText + "\n";
          }
          fullText = fullText.trim();
        } catch (pdfError) {
          addLog(`Error al leer el PDF con pdfjs: ${pdfError}`);
          fullText = "";
        }
        if (stale()) return;
        // Un PDF escaneado suele devolver nada (o casi nada) de texto:
        // en ese caso recurrimos a OCR con Gemini.
        if (fullText.length >= MIN_PDF_TEXT_CHARS) {
          setCvText(fullText);
        } else if (apiKey.trim()) {
          addLog("PDF sin capa de texto, usando OCR con Gemini...");
          await runOcr();
        } else {
          setNeedsOcr(true);
          setFileError(String(t.cv.errors.pdfScanned));
        }
      } else if (isImage) {
        addLog("Procesando imagen del CV...");
        const imageBytes = await uploadedFile.arrayBuffer();
        if (stale()) return;
        docRef.current = {
          bytes: imageBytes,
          mimeType: uploadedFile.type || "image/png",
        };
        // Las imágenes siempre pasan por OCR con Gemini
        if (apiKey.trim()) {
          await runOcr();
        } else {
          setNeedsOcr(true);
          setFileError(String(t.cv.errors.imageNeedsOcr));
        }
      } else if (isDocx) {
        addLog("Procesando archivo DOCX...");
        const mammoth = await import("mammoth");
        const result = await mammoth.extractRawText({
          arrayBuffer: await uploadedFile.arrayBuffer(),
        });
        const docxText = result.value.trim();
        if (stale()) return;
        if (!docxText) {
          setFile(null);
          setFileError(String(t.cv.errors.docxEmpty));
          return;
        }
        setCvText(docxText);
      } else if (isText) {
        addLog("Procesando archivo TXT...");
        const fileText = (await uploadedFile.text()).trim();
        if (stale()) return;
        if (!fileText) {
          setFile(null);
          setFileError(String(t.cv.errors.txtEmpty));
          return;
        }
        setCvText(fileText);
      }
    } catch (error) {
      addLog(`Error procesando archivo: ${error}`);
      console.error("Error procesando archivo:", error);
      if (uploadIdRef.current === uploadId) {
        setFile(null);
        setFileError(
          String(
            t.cv.errors.processFailed({
              reason:
                error instanceof Error ? error.message : String(t.api.unknown),
            })
          )
        );
      }
    } finally {
      setIsExtracting(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFileUpload(dropped);
  };

  // ── Paso 3: libreto ────────────────────────────────────────────────

  const generateEpisode = async () => {
    if (scriptRunningRef.current || busy || isExportingVideo) return;
    if (!hasKey || !hasCv) return;
    scriptRunningRef.current = true;

    addLog("Iniciando generación de episodio...");
    setScriptError("");
    setAudioError("");
    setPodcastScript("");
    resetAudioState();
    setIsGeneratingScript(true);

    try {
      const ai = await createClient(apiKey);
      // El prompt vive en i18n/messages/*.ts (sección `prompt`): el episodio se
      // genera en el idioma activo de la UI. Interpolamos la fecha actual para
      // que el modelo no vea "inconsistencias temporales" fantasma. La
      // formateamos en calendario gregoriano con año en dígitos latinos (el mes
      // sí en el idioma) para que sea comparable con las fechas del CV: sin esto
      // fa saldría en Jalali y th en calendario budista. intl-t deja pasar los
      // strings tal cual (solo auto-formatea objetos Date).
      const dateStr = new Intl.DateTimeFormat(locale, {
        calendar: "gregory",
        numberingSystem: "latn",
        dateStyle: "long",
      }).format(new Date());
      const scriptPrompt = String(t.prompt.script({ date: dateStr }));

      let script = "";
      if (devMode) {
        script = cvText;
        setPodcastScript(script);
      } else {
        // Si tenemos el documento original (PDF o imagen), el modelo también
        // puede "verlo": foto, diseño, tipografía y vibe general como material
        // extra para el humor, sin tomarlo demasiado literal.
        const doc = docRef.current;
        const vibesNote = doc ? `\n\n${t.prompt.vibes}` : "";
        const parts: Part[] = [
          { text: `${scriptPrompt}${vibesNote}\n\n${cvText}` },
        ];
        if (doc) {
          addLog("Adjuntando documento original para contexto visual...");
          parts.push({
            inlineData: {
              mimeType: doc.mimeType,
              data: arrayBufferToBase64(doc.bytes),
            },
          });
        }
        const { result: generatedScript } = await callWithFallback(
          SCRIPT_MODELS,
          async (model) => {
            setActivityNote(String(t.episode.progress.writingWith({ model })));
            setPodcastScript(""); // limpiar texto parcial de intentos previos
            let accumulated = "";
            // El stream llega a más velocidad de la que vale la pena
            // renderizar: volcamos como mucho una vez por frame
            let flushScheduled = false;
            const flush = () => {
              flushScheduled = false;
              setPodcastScript(accumulated);
            };
            const result = await ai.models.generateContentStream({
              model,
              contents: [{ parts }],
            });
            for await (const chunk of result) {
              accumulated += chunk.text || "";
              if (!flushScheduled) {
                flushScheduled = true;
                requestAnimationFrame(flush);
              }
            }
            setPodcastScript(accumulated);
            return accumulated;
          },
          (message) => {
            addLog(message);
            setActivityNote(message);
          }
        );
        script = generatedScript;
      }

      if (!script.trim()) {
        throw new Error(String(t.episode.errors.scriptEmpty));
      }
      addLog(`Episodio generado: ${script.substring(0, 100)}...`);
      setPodcastScript(script);
      setIsGeneratingScript(false);
      scriptRunningRef.current = false;

      await generateAudio(script, { resume: false });
    } catch (error) {
      addLog(`Error generando episodio: ${error}`);
      console.error("Error generando episodio:", error);
      setPodcastScript("");
      setScriptError(
        String(
          t.episode.errors.scriptFailed({ reason: apiErrorMessage(error) })
        )
      );
    } finally {
      setIsGeneratingScript(false);
      setActivityNote("");
      scriptRunningRef.current = false;
    }
  };

  const copyScript = async () => {
    try {
      await navigator.clipboard.writeText(podcastScript);
      setCopyState("copied");
      setTimeout(() => setCopyState("idle"), 2000);
    } catch {
      setScriptError(String(t.episode.errors.copyFailed));
    }
  };

  // ── Paso 4: audio ──────────────────────────────────────────────────

  const resetAudioState = () => {
    setAudioSegments([]);
    setTotalSegments(0);
    setFullAudio(null);
    setAudioError("");
    setSegmentTexts([]);
    pcmChunksRef.current = [];
    segTextsRef.current = [];
    ttsModelIdxRef.current = 0;
  };

  // Genera el audio con Gemini TTS por segmentos. Con { resume: true }
  // continúa desde el último segmento completado sin descartar lo generado.
  const generateAudio = async (
    script: string,
    { resume }: { resume: boolean }
  ) => {
    if (audioRunningRef.current) return;
    audioRunningRef.current = true;
    setAudioError("");
    setIsGeneratingAudio(true);
    try {
      const ai = await createClient(apiKey);
      const stylePrefix = devMode ? "" : `${t.prompt.ttsStyle}\n`;

      if (!resume || segTextsRef.current.length === 0) {
        resetAudioState();
        segTextsRef.current = splitScriptIntoSegments(
          script,
          TTS_SEGMENT_MAX_CHARS,
          SPEAKERS.map((speaker) => speaker.name)
        );
        setTotalSegments(segTextsRef.current.length);
        setSegmentTexts(segTextsRef.current);
      }
      const segTexts = segTextsRef.current;
      addLog(
        `Generando audio: ${segTexts.length} segmento(s), desde el ${
          pcmChunksRef.current.length + 1
        }...`
      );

      for (let i = pcmChunksRef.current.length; i < segTexts.length; i++) {
        setActivityNote(
          String(
            t.episode.progress.recordingPart({
              current: i + 1,
              total: segTexts.length,
            })
          )
        );
        const { result, modelIdx } = await callWithFallback(
          TTS_MODELS,
          async (ttsModel) => {
            const ttsResult = await ai.models.generateContent({
              model: ttsModel,
              contents: [{ parts: [{ text: stylePrefix + segTexts[i] }] }],
              config: {
                responseModalities: ["AUDIO"],
                speechConfig: {
                  multiSpeakerVoiceConfig: {
                    speakerVoiceConfigs: SPEAKERS.map((speaker) => ({
                      speaker: speaker.name,
                      voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: speaker.voice },
                      },
                    })),
                  },
                },
              },
            });
            // El audio puede llegar repartido en varias partes
            const audioParts =
              ttsResult.candidates?.[0]?.content?.parts?.filter(
                (part) => part.inlineData?.data
              ) ?? [];
            if (audioParts.length === 0) {
              throw new Error(String(t.episode.errors.noAudio));
            }
            const pcm = concatPcm(
              audioParts.map((part) =>
                base64ToUint8Array(String(part.inlineData!.data))
              )
            );
            return {
              pcm,
              rate: parseSampleRate(audioParts[0].inlineData?.mimeType),
            };
          },
          (message) => {
            addLog(message);
            setActivityNote(message);
          },
          ttsModelIdxRef.current
        );
        ttsModelIdxRef.current = modelIdx;
        if (
          pcmChunksRef.current.length > 0 &&
          result.rate !== sampleRateRef.current
        ) {
          // Latente: hoy todos los modelos TTS emiten 24 kHz, pero si un
          // fallback cambiara el rate a mitad de episodio, el WAV final
          // sonaría mal. Dejamos constancia para poder diagnosticarlo.
          addLog(
            `⚠️ Sample rate distinto entre segmentos (${sampleRateRef.current} → ${result.rate} Hz)`
          );
        }
        sampleRateRef.current = result.rate;
        setSampleRate(result.rate);
        pcmChunksRef.current.push(result.pcm);
        setAudioSegments((prev) => [
          ...prev,
          pcmToWav(result.pcm, result.rate),
        ]);
      }

      const fullPcm = concatPcm(pcmChunksRef.current);
      addLog(
        `Audio completo: ${fullPcm.length} bytes PCM @ ${sampleRateRef.current} Hz`
      );
      setFullAudio(pcmToWav(fullPcm, sampleRateRef.current));
    } catch (error) {
      addLog(`Error generando audio: ${error}`);
      console.error("Error generando audio:", error);
      setAudioError(
        String(t.episode.errors.audioFailed({ reason: apiErrorMessage(error) }))
      );
    } finally {
      setIsGeneratingAudio(false);
      setActivityNote("");
      audioRunningRef.current = false;
    }
  };

  // ── Paso 5: exportar y compartir ───────────────────────────────────

  const episodeFileName = () => {
    const name = file?.name || "episodio";
    const dot = name.lastIndexOf(".");
    const base = dot > 0 ? name.slice(0, dot) : name;
    return `the-cv-comedy-podcast-${base}`;
  };

  const downloadScript = () => {
    if (!podcastScript) return;
    downloadBlob(
      new Blob([podcastScript], { type: "text/plain" }),
      `${episodeFileName()}.txt`
    );
  };

  const downloadAudio = () => {
    if (fullAudio) downloadBlob(fullAudio, `${episodeFileName()}.wav`);
  };

  // Exporta el episodio como video (portada + visualizador + audio).
  // La grabación ocurre en tiempo real: dura lo mismo que el episodio.
  const exportVideo = async () => {
    if (!fullAudio || isExportingVideo) return;
    setIsExportingVideo(true);
    setVideoProgress(0);
    setVideoError("");
    const abort = new AbortController();
    videoAbortRef.current = abort;
    try {
      addLog("Exportando video del episodio (en tiempo real)...");
      const videoBlob = await exportEpisodeVideo(
        fullAudio,
        "/cover.png",
        (fraction) => setVideoProgress(fraction * 100),
        abort.signal
      );
      const ext = videoBlob.type.includes("mp4") ? "mp4" : "webm";
      downloadBlob(videoBlob, `${episodeFileName()}.${ext}`);
      addLog(
        `Video exportado (${(videoBlob.size / 1024 / 1024).toFixed(1)} MB)`
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        addLog("Exportación de video cancelada");
      } else {
        addLog(`Error exportando video: ${error}`);
        setVideoError(
          String(
            t.episode.errors.videoFailed({
              reason:
                error instanceof Error ? error.message : String(t.api.unknown),
            })
          )
        );
      }
    } finally {
      setIsExportingVideo(false);
      videoAbortRef.current = null;
    }
  };

  const shareEpisode = async () => {
    const shareData = {
      title: "The CV Comedy Podcast",
      text: String(t.episode.shareText),
      url: APP_URL,
    };
    try {
      const audioFile = fullAudio
        ? new File([fullAudio], `${episodeFileName()}.wav`, {
            type: "audio/wav",
          })
        : null;
      if (
        audioFile &&
        typeof navigator.canShare === "function" &&
        navigator.canShare({ files: [audioFile] })
      ) {
        await navigator.share({ ...shareData, files: [audioFile] });
      } else if (navigator.share) {
        await navigator.share(shareData);
      } else {
        await navigator.clipboard.writeText(
          `${shareData.text} ${shareData.url}`
        );
        setShareState("copied");
        setTimeout(() => setShareState("idle"), 2000);
      }
    } catch {
      // usuario canceló el diálogo de compartir: no es un error
    }
  };

  const startNewEpisode = () => {
    if (busy || isExportingVideo) return;
    setPodcastScript("");
    setScriptError("");
    resetAudioState();
    setVideoError("");
    setVideoProgress(0);
  };

  // ── Render ─────────────────────────────────────────────────────────

  const generateDisabledReason = !hasKey
    ? t.episode.missingKey
    : !hasCv
      ? t.episode.missingCv
      : isExportingVideo
        ? t.episode.waitVideo
        : "";

  // SEO i18n: canonical por idioma + Open Graph. Las alternantes hreflang
  // (todas las URLs por locale + x-default → la raíz) van en el <Head>.
  const canonical = localeUrl(locale);
  const ogImage = `${APP_URL}cover-og.jpg`;
  // Items de features (contenido explicativo + featureList del schema)
  const featureItems = [
    t.features.items.formats,
    t.features.items.script,
    t.features.items.voices,
    t.features.items.streaming,
    t.features.items.download,
    t.features.items.privacy,
  ];
  const jsonLd = JSON.stringify({
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${SITE_URL}/#website`,
        url: SITE_URL,
        name: "The CV Comedy Podcast",
        inLanguage: [...LOCALES],
      },
      {
        "@type": "WebApplication",
        "@id": `${SITE_URL}/#app`,
        name: "The CV Comedy Podcast",
        url: SITE_URL,
        applicationCategory: "MultimediaApplication",
        operatingSystem: "Web",
        inLanguage: [...LOCALES],
        description: String(t.meta.description),
        featureList: featureItems.map((item) => String(item.title)),
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        isPartOf: { "@id": `${SITE_URL}/#website` },
      },
    ],
  });

  return (
    <>
      <Head>
        <title>{String(t.meta.title)}</title>
        <meta name="description" content={String(t.meta.description)} />
        <meta name="keywords" content={String(t.meta.keywords)} />
        <meta
          name="robots"
          content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1"
        />
        <link rel="canonical" href={canonical} />
        {/* Alternantes por idioma (crawlables) + x-default = raíz que negocia */}
        <link rel="alternate" hrefLang="x-default" href={`${SITE_URL}/`} />
        {LOCALES.map((loc) => (
          <link
            key={`alt-${loc}`}
            rel="alternate"
            hrefLang={loc}
            href={localeUrl(loc)}
          />
        ))}
        <meta property="og:title" content={String(t.meta.ogTitle)} />
        <meta
          property="og:description"
          content={String(t.meta.ogDescription)}
        />
        <meta property="og:image" content={ogImage} />
        <meta property="og:image:width" content="1024" />
        <meta property="og:image:height" content="1024" />
        <meta property="og:image:alt" content="The CV Comedy Podcast" />
        <meta property="og:url" content={canonical} />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="The CV Comedy Podcast" />
        <meta property="og:locale" content={OG_LOCALES[locale]} />
        {LOCALES.filter((loc) => loc !== locale).map((loc) => (
          <meta
            key={`ogloc-${loc}`}
            property="og:locale:alternate"
            content={OG_LOCALES[loc]}
          />
        ))}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="The CV Comedy Podcast" />
        <meta
          name="twitter:description"
          content={String(t.meta.ogDescription)}
        />
        <meta name="twitter:image" content={ogImage} />
        <meta name="twitter:image:alt" content="The CV Comedy Podcast" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: jsonLd }}
        />
      </Head>
      <div className="min-h-screen bg-linear-to-br from-purple-50 to-blue-50 px-4 py-3 transition-colors sm:py-4 dark:from-zinc-950 dark:to-zinc-900">
        <div className="mx-auto max-w-3xl">
          {/* Skip link: primer elemento enfocable, visible solo al recibir foco */}
          <a
            href="#main"
            className="sr-only rounded-lg bg-purple-600 px-4 py-2 text-sm font-semibold text-white focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50"
          >
            {t.a11y.skip}
          </a>
          {/* Header: el toggle va en flujo normal (nunca se superpone) */}
          <header className="mb-8">
            <div className="flex items-center justify-end gap-2">
              <LanguageSwitcher />
              <ThemeToggle />
            </div>
            <div className="mt-1 text-center sm:mt-0">
              <h1 className="mb-2 text-3xl font-bold text-gray-800 sm:text-4xl dark:text-zinc-100">
                <span aria-hidden="true">🎙️</span> The CV Comedy Podcast
              </h1>
              <p className="mb-4 text-base text-zinc-600 sm:text-lg dark:text-zinc-300">
                {t.header.tagline}
              </p>
            </div>
            {/* Los badges SON el disparador (details/summary nativo y accesible):
                al hacer click se despliega debajo, discreto, el contenido
                explicativo — que va SIEMPRE en el DOM (indexable) aunque cerrado.
                Un chevron tenue insinúa que es expandible; la pista textual
                queda solo para lectores de pantalla (sr-only). */}
            <details className="group">
              <summary className="mx-auto flex w-fit cursor-pointer list-none flex-wrap items-center justify-center gap-2 rounded-full text-xs font-medium select-none marker:content-none focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-purple-500 [&::-webkit-details-marker]:hidden">
                <span className="sr-only">{t.features.toggle}</span>
                <span className="rounded-full bg-green-100 px-3 py-1 text-green-800 dark:bg-green-950 dark:text-green-300">
                  {t.header.badges.tts}
                </span>
                <span className="rounded-full bg-blue-100 px-3 py-1 text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                  {t.header.badges.gemini}
                </span>
                <span className="rounded-full bg-purple-100 px-3 py-1 text-purple-800 dark:bg-purple-950 dark:text-purple-300">
                  {t.header.badges.humor}
                </span>
                <svg
                  aria-hidden="true"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="h-3.5 w-3.5 shrink-0 text-zinc-300 transition-transform group-open:rotate-180 dark:text-zinc-600"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.17l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                    clipRule="evenodd"
                  />
                </svg>
              </summary>
              <div className="mx-auto mt-4 max-w-xl border-t border-zinc-200/70 pt-4 text-start dark:border-zinc-800/60">
                <h2
                  id="features-heading"
                  className="text-sm font-semibold text-zinc-700 dark:text-zinc-200"
                >
                  {t.features.heading}
                </h2>
                <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                  {t.features.intro}
                </p>
                <dl className="mt-3 grid gap-x-5 gap-y-3 sm:grid-cols-2">
                  {featureItems.map((item, i) => (
                    <div key={i}>
                      <dt className="text-sm font-semibold text-zinc-600 dark:text-zinc-300">
                        {item.title}
                      </dt>
                      <dd className="text-sm text-zinc-500 dark:text-zinc-400">
                        {item.desc}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            </details>
          </header>

          <main id="main" className="flex flex-col gap-4">
            {/* Paso 1: API Key */}
            <StepCard
              number={1}
              title={String(t.apikey.title)}
              status={stepStatus[1]}
            >
              <label
                htmlFor="apiKey"
                className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300"
              >
                {t.apikey.inputLabel}
              </label>
              <input
                type="password"
                id="apiKey"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={String(t.apikey.placeholder)}
                autoComplete="off"
                spellCheck={false}
                className="w-full rounded-xl border border-zinc-300 bg-white p-3 text-zinc-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
              />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                <label className="flex cursor-pointer items-center gap-2 py-2 text-sm text-zinc-600 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={rememberKey}
                    onChange={(e) => setRememberKey(e.target.checked)}
                    className="h-4 w-4 accent-purple-600"
                  />
                  {t.apikey.remember}
                </label>
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="py-2 text-sm text-blue-600 hover:underline dark:text-blue-300"
                >
                  {t.apikey.getKey}
                </a>
              </div>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                {t.apikey.note}
              </p>
            </StepCard>

            {/* Paso 2: CV */}
            <StepCard
              number={2}
              title={String(t.cv.title)}
              status={stepStatus[2]}
            >
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                disabled={busy || isExtracting || isOcrRunning}
                className={`w-full rounded-xl border-2 border-dashed p-6 text-center transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-purple-500 disabled:opacity-50 ${
                  dragActive
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950"
                    : "border-zinc-300 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-500"
                }`}
              >
                {file ? (
                  <span className="flex flex-col items-center gap-1">
                    <span className="font-medium text-green-700 dark:text-green-400">
                      ✅ {file.name}
                    </span>
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {t.cv.sizeReplace({
                        size: (file.size / 1024).toFixed(1),
                      })}
                    </span>
                  </span>
                ) : (
                  <span className="flex flex-col items-center gap-1">
                    <span className="text-zinc-700 dark:text-zinc-200">
                      {t.cv.dropDrag}
                    </span>
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {t.cv.dropFormats}
                    </span>
                  </span>
                )}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".pdf,.txt,.docx,.png,.jpg,.jpeg,.webp"
                onChange={(e) => {
                  const selected = e.target.files?.[0];
                  if (selected) handleFileUpload(selected);
                  e.target.value = ""; // permite volver a subir el mismo archivo
                }}
              />

              {(isExtracting || isOcrRunning) && (
                <p
                  aria-live="polite"
                  className="mt-3 flex items-center justify-center gap-2 text-sm font-medium text-purple-700 dark:text-purple-300"
                >
                  <Spinner />
                  {isOcrRunning ? t.cv.ocring : t.cv.processing}
                </p>
              )}

              {showOcrButton && (
                <div className="mt-3 text-center">
                  <Button
                    variant="secondary"
                    onClick={runOcr}
                    disabled={!hasKey || isOcrRunning}
                  >
                    {t.cv.ocrButton}
                  </Button>
                </div>
              )}

              {fileError && <Alert>{fileError}</Alert>}

              <div className="mt-3 flex items-center justify-between gap-2">
                <label
                  htmlFor="cvText"
                  className="text-sm font-medium text-zinc-700 dark:text-zinc-300"
                >
                  {t.cv.textLabel}
                </label>
                {file && (
                  <Button variant="ghost" onClick={clearFile} disabled={busy}>
                    {t.cv.clearFile}
                  </Button>
                )}
              </div>
              <textarea
                id="cvText"
                value={cvText}
                onChange={(e) => setCvText(e.target.value)}
                placeholder={String(
                  file ? t.cv.placeholderFile : t.cv.placeholderManual
                )}
                disabled={isExtracting || busy}
                className="mt-1 h-36 w-full resize-none rounded-xl border border-zinc-300 bg-white p-3 text-sm text-zinc-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
              />
            </StepCard>

            {/* Paso 3: El episodio (libreto + audio + compartir) */}
            <StepCard
              number={3}
              title={String(t.episode.title)}
              status={stepStatus[3]}
            >
              <Button
                variant="primary"
                className="w-full"
                onClick={generateEpisode}
                disabled={busy || !hasKey || !hasCv || isExportingVideo}
              >
                {busy ? (
                  <>
                    <Spinner /> {t.episode.generating}
                  </>
                ) : scriptReady ? (
                  t.episode.regenerate
                ) : (
                  t.episode.generate
                )}
              </Button>
              {!busy && !isExportingVideo && generateDisabledReason && (
                <p
                  aria-live="polite"
                  className="mt-2 text-center text-xs text-zinc-500 dark:text-zinc-400"
                >
                  {generateDisabledReason}
                </p>
              )}

              {/* Una única barra de progreso para todo: libreto → audio → video */}
              {pipeline.show && (
                <div className="mt-3">
                  <ProgressBar
                    value={pipeline.value}
                    indeterminate={pipeline.indeterminate}
                    label={pipeline.label}
                  />
                </div>
              )}

              {scriptError && <Alert>{scriptError}</Alert>}

              {podcastScript && (
                <div className="mt-4 flex flex-col gap-4 md:flex-row">
                  {/* Libreto a la izquierda */}
                  <div className="min-w-0 flex-1">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                        {isGeneratingScript
                          ? t.episode.scriptWriting
                          : t.episode.scriptReady}
                      </h3>
                      {!isGeneratingScript && (
                        <Button variant="ghost" size="sm" onClick={copyScript}>
                          {copyState === "copied"
                            ? t.episode.copied
                            : t.episode.copy}
                        </Button>
                      )}
                    </div>
                    <div
                      aria-busy={isGeneratingScript}
                      className="max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-zinc-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-zinc-200"
                    >
                      {podcastScript}
                    </div>
                  </div>
                  {/* Reproductor a la derecha */}
                  {(audioStarted || audioReady) && (
                    <div className="min-w-0 flex-1">
                      <EpisodePlayer
                        segments={audioSegments}
                        fullAudio={fullAudio}
                        totalSegments={totalSegments}
                        segmentTexts={segmentTexts}
                        speakerNames={SPEAKER_NAMES}
                        isGenerating={isGeneratingAudio}
                        sampleRate={sampleRate}
                      />
                    </div>
                  )}
                </div>
              )}

              {audioError && (
                <>
                  <Alert>{audioError}</Alert>
                  <div className="mt-3 text-center">
                    <Button
                      variant="secondary"
                      onClick={() =>
                        generateAudio(podcastScript, { resume: true })
                      }
                      disabled={busy}
                    >
                      {audioSegments.length > 0
                        ? t.episode.errors.resume({
                            part: audioSegments.length + 1,
                          })
                        : t.episode.errors.retryAudio}
                    </Button>
                  </div>
                </>
              )}

              {/* Compartir y descargar, en la misma tarjeta */}
              {scriptReady && (
                <div className="mt-4 flex flex-wrap items-center justify-center gap-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={shareEpisode}
                    disabled={!audioReady}
                  >
                    {shareState === "copied"
                      ? t.episode.linkCopied
                      : t.episode.share}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={downloadAudio}
                    disabled={!audioReady}
                  >
                    {t.episode.audioFile}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={downloadScript}
                  >
                    {t.episode.scriptFile}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={
                      isExportingVideo
                        ? () => videoAbortRef.current?.abort()
                        : exportVideo
                    }
                    disabled={!audioReady && !isExportingVideo}
                  >
                    {isExportingVideo ? t.episode.cancelVideo : t.episode.video}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={startNewEpisode}
                    disabled={busy || isExportingVideo}
                  >
                    {t.episode.newEpisode}
                  </Button>
                </div>
              )}
              {videoError && <Alert>{videoError}</Alert>}
            </StepCard>

            {/* Apoyo: GitHub Sponsors */}
            {audioReady && (
              <section
                aria-labelledby="sponsor-heading"
                className="rounded-2xl border border-pink-200 bg-white p-5 text-center shadow-sm dark:border-pink-900 dark:bg-zinc-900"
              >
                <h2
                  id="sponsor-heading"
                  className="mb-1 text-lg font-semibold text-zinc-800 dark:text-zinc-100"
                >
                  {t.sponsor.title}
                </h2>
                <p className="mx-auto mb-3 max-w-md text-sm text-zinc-600 dark:text-zinc-300">
                  {t.sponsor.text}
                </p>
                <a
                  href="https://github.com/sponsors/nivandres"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-xl border border-pink-300 bg-pink-50 px-5 py-2.5 text-sm font-semibold text-pink-800 transition-colors hover:bg-pink-100 dark:border-pink-800 dark:bg-pink-950 dark:text-pink-200 dark:hover:bg-pink-900"
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                    className="h-4 w-4"
                  >
                    <path d="M4.25 2.5c-1.336 0-2.75 1.164-2.75 3 0 2.15 1.58 4.144 3.365 5.682A20.565 20.565 0 0 0 8 13.393a20.561 20.561 0 0 0 3.135-2.211C12.92 9.644 14.5 7.65 14.5 5.5c0-1.836-1.414-3-2.75-3-1.373 0-2.609.986-3.029 2.456a.749.749 0 0 1-1.442 0C6.859 3.486 5.623 2.5 4.25 2.5Z" />
                  </svg>
                  {t.sponsor.button}
                </a>
              </section>
            )}

            {/* Logs de depuración (solo ?dev=1) */}
            {devMode && debugLog.length > 0 && (
              <div className="max-h-48 overflow-y-auto rounded-2xl bg-zinc-100 p-4 font-mono text-xs text-zinc-600 dark:bg-zinc-900 dark:text-zinc-400">
                <h3 className="mb-2 font-semibold">Logs de depuración:</h3>
                {debugLog.map((log, idx) => (
                  <div key={idx}>{log}</div>
                ))}
              </div>
            )}
          </main>

          {/* Footer */}
          <footer className="mt-8 mb-6 flex flex-col items-center gap-1 text-center text-sm text-gray-500 dark:text-zinc-400">
            <p>{t.footer.disclaimer}</p>
            <a
              href="https://github.com/nivandres/the-cv-comedy-podcast"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-flex items-center gap-2 rounded-lg bg-gray-900 px-4 py-2 text-base font-semibold text-white shadow transition-colors hover:bg-gray-800 dark:bg-zinc-800 dark:hover:bg-zinc-700"
            >
              <svg
                className="h-5 w-5"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.729.083-.729 1.205.085 1.84 1.237 1.84 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.775.418-1.305.762-1.606-2.665-.304-5.466-1.334-5.466-5.931 0-1.31.468-2.381 1.236-3.221-.124-.303-.535-1.523.117-3.176 0 0 1.008-.322 3.3 1.23.957-.266 1.984-.399 3.003-.404 1.018.005 2.046.138 3.006.404 2.289-1.552 3.295-1.23 3.295-1.23.653 1.653.242 2.873.119 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.804 5.625-5.475 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 0 .322.218.694.825.576C20.565 21.796 24 17.299 24 12c0-6.627-5.373-12-12-12z" />
              </svg>
              {t.footer.repo}
            </a>
          </footer>
        </div>
      </div>
    </>
  );
}

// Cada idioma se PRE-GENERA como página estática (ISR): el proxy de intl-t
// (strategy "param") rutea la raíz "/" al idioma detectado por cookie y sirve
// /es /en /ja … como URLs canónicas crawlables, todas servidas desde HTML
// estático (TTFB mínimo, mejor Core Web Vitals) y revalidadas periódicamente.
export const getStaticPaths: GetStaticPaths = async () => ({
  paths: LOCALES.map((locale) => ({ params: { locale } })),
  fallback: false,
});

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const raw = params?.locale;
  const locale = isAppLocale(raw) ? raw : DEFAULT_LOCALE;
  const messages = await localeLoaders[locale]();
  // revalidate: el contenido es estático, pero ISR mantiene las páginas frescas
  return { props: { locale, messages }, revalidate: 3600 };
};
