import { useState, useRef, useEffect } from "react";
import Head from "next/head";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleGenAI } from "@google/genai";

// Helper to convert base64 to Blob URL
function base64ToBlob(base64: string, mimeType: string) {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: mimeType });
  return blob;
}

// Helper to wrap PCM in WAV header (for Gemini TTS output)
function pcmToWav(
  pcm: Uint8Array,
  sampleRate = 24000,
  numChannels = 1,
  bitsPerSample = 16
) {
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const wavBuffer = new ArrayBuffer(44 + pcm.length);
  const view = new DataView(wavBuffer);

  // RIFF identifier 'RIFF'
  view.setUint32(0, 0x52494646, false);
  // file length minus RIFF identifier length and file description length
  view.setUint32(4, 36 + pcm.length, true);
  // RIFF type 'WAVE'
  view.setUint32(8, 0x57415645, false);
  // format chunk identifier 'fmt '
  view.setUint32(12, 0x666d7420, false);
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw)
  view.setUint16(20, 1, true);
  // channel count
  view.setUint16(22, numChannels, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, byteRate, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, blockAlign, true);
  // bits per sample
  view.setUint16(34, bitsPerSample, true);
  // data chunk identifier 'data'
  view.setUint32(36, 0x64617461, false);
  // data chunk length
  view.setUint32(40, pcm.length, true);

  // Write PCM samples
  new Uint8Array(wavBuffer, 44).set(pcm);

  return new Blob([wavBuffer], { type: "audio/wav" });
}

declare global {
  interface Window {
    paypal: any;
  }
}

interface Speaker {
  name: string;
  voice: string;
  personality: string;
}

// @ts-ignore
"localStorage" in globalThis ? null : (globalThis.localStorage = null);

// Simple ProgressBar component for visual feedback
function ProgressBar({ duration }: { duration: number }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    setProgress(0);
    if (duration <= 0) return;
    const start = Date.now();
    const interval = setInterval(() => {
      const elapsed = (Date.now() - start) / 1000;
      setProgress(Math.min((elapsed / duration) * 100, 100));
      if (elapsed >= duration) clearInterval(interval);
    }, 100);
    return () => clearInterval(interval);
  }, [duration]);

  return (
    <div className="w-full bg-gray-200 rounded-full h-6 shadow-md border border-gray-300 flex items-center">
      <div
        className="h-6 rounded-full transition-all flex items-center justify-center text-white font-bold text-sm shadow-lg"
        style={{
          width: `${progress}%`,
          background:
            "linear-gradient(90deg, #f72585 0%, #b5179e 50%, #ff8800 100%)",
          boxShadow: "0 2px 8px 0 rgba(247,37,133,0.2)",
          transition: "width 0.2s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        {progress > 10 && (
          <span className="ml-2 text-xs font-semibold drop-shadow">
            {Math.round(progress)}%
          </span>
        )}
      </div>
    </div>
  );
}

// AudioVisualizer: visualizador de barras sobre imagen de fondo
function AudioVisualizer({ audioBuffer }: { audioBuffer: Blob }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    const canvas = canvasRef.current;
    if (!audio || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let animationId: number;
    let analyser: AnalyserNode | null = null;
    let dataArray: Uint8Array;
    let source: MediaElementAudioSourceNode | null = null;
    let audioCtx: AudioContext | null = null;

    function draw() {
      if (!analyser || !ctx) return;
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, canvas!.width, canvas!.height);
      // Fondo transparente para ver la portada
      const barWidth = (canvas!.width / dataArray.length) * 2.5;
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = dataArray[i] * 0.7;
        const grad = ctx.createLinearGradient(0, 0, 0, canvas!.height);
        grad.addColorStop(0, "#f72585");
        grad.addColorStop(0.5, "#b5179e");
        grad.addColorStop(1, "#ff8800");
        ctx.fillStyle = grad;
        ctx.globalAlpha = 0.5; // Opacidad media
        ctx.fillRect(x, canvas!.height - barHeight, barWidth, barHeight);
        ctx.globalAlpha = 1.0;
        x += barWidth + 1;
      }
      animationId = requestAnimationFrame(draw);
    }

    function setupAudio() {
      audioCtx = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      dataArray = new Uint8Array(analyser.frequencyBinCount);
      if (audio) {
        source = audioCtx.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
      }
    }
    setupAudio();
    draw();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (audioCtx) audioCtx.close();
    };
  }, [audioBuffer]);

  return (
    <div className="relative w-full flex flex-col items-center">
      <div className="relative w-full max-w-xl aspect-square rounded-lg overflow-hidden shadow-lg">
        <img
          src="/cover.png"
          alt="Portada del episodio"
          className="absolute inset-0 w-full h-full object-cover z-0"
        />
        <canvas
          ref={canvasRef}
          width={800}
          height={800}
          className="absolute inset-0 w-full h-full z-10 pointer-events-none opacity-75"
        />
      </div>
      <audio
        ref={audioRef}
        src={URL.createObjectURL(audioBuffer)}
        className="w-full max-w-xl mt-2"
        controls
      />
    </div>
  );
}

export default function TheCVComedyPodcast() {
  const [apiKey, setApiKey] = useState<string>(
    localStorage?.getItem("apiKey") || ""
  );
  const [file, setFile] = useState<File | null>(null);
  const [manualText, setManualText] = useState<string>("");
  const [podcastScript, setPodcastScript] = useState<string>("");
  const [audioBuffer, setAudioBuffer] = useState<Blob | null>(null);
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  let [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [debugLog, setDebugLog] = useState<string[]>([]);
  const [isGeneratingScript, setIsGeneratingScript] = useState<boolean>(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState<boolean>(false);
  const [speakers] = useState<Speaker[]>([
    {
      name: "Alex",
      voice: "Kore",
      personality:
        "Analítico y sarcástico, hace observaciones técnicas precisas",
    },
    {
      name: "Sam",
      voice: "Puck",
      personality:
        "Espontáneo y gracioso, hace comentarios divertidos y observaciones casuales",
    },
  ]);
  const [showManualInput, setShowManualInput] = useState<boolean>(false);

  // Mockup de resumen de CV para modo dev
  const mockCvSummary = `NOMBRE DEL CANDIDATO: Juan Pérez\nPROFESIÓN/ÁREA: Ingeniero de Software\nEXPERIENCIA PRINCIPAL:\n- 5 años desarrollando aplicaciones web\n- Experiencia liderando equipos ágiles\nEDUCACIÓN: Licenciatura en Ingeniería Informática\nHABILIDADES DESTACADAS: JavaScript, React, Node.js, liderazgo\nOBSERVACIONES GENERALES: CV bien estructurado, pero usa muchos clichés y frases genéricas.`;

  // --- Modo dev para mockup de audio ---
  const [devMode, setDevMode] = useState<boolean>(false);
  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      window.location.search.includes("dev=1")
    ) {
      setDevMode(true);
    }
  }, []);

  // Precarga automática de mock audio y script en modo dev
  useEffect(() => {
    if (devMode) {
      setPodcastScript(
        "[Episodio de prueba]\nAlex: Bienvenidos a The CV Comedy Podcast.\nSam: Hoy solo estamos probando el visualizador de audio.\n[...]"
      );
      setManualText(mockCvSummary); // Mostrar el mock summary en el área manual
      // Solo cargar audio si aún no está cargado
      if (!audioBuffer) {
        (async () => {
          try {
            const response = await fetch(
              "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            );
            const blob = await response.blob();
            setAudioBuffer(blob);
          } catch (e) {
            setError("No se pudo cargar el audio de prueba");
          }
        })();
      }
    }
  }, [devMode]);
  // --- Fin modo dev ---

  isGenerating ||= isGeneratingScript || isGeneratingAudio;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const addLog = (message: string) => {
    console.log(message);
    setDebugLog((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()}: ${message}`,
    ]);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = e.dataTransfer.files;
    handleFileUpload(files[0]);
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(",")[1]);
      };
      reader.onerror = (error) => reject(error);
    });
  };

  const handleFileUpload = async (uploadedFile: File) => {
    if (!uploadedFile) {
      setError("No se pudo cargar el archivo");
      return;
    }
    addLog(`Iniciando carga de archivo: ${uploadedFile.name}`);
    setFile(uploadedFile);
    setIsExtracting(true);
    setError("");
    setManualText(""); // Limpiar manualText al subir nuevo archivo

    try {
      if (uploadedFile.type === "application/pdf") {
        addLog("Procesando archivo PDF...");
        setManualText("");
        try {
          const pdfjsLib: any = await import("pdfjs-dist");
          pdfjsLib.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@latest/build/pdf.worker.min.mjs`;
          const arrayBuffer = await uploadedFile.arrayBuffer();
          const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
          const pdf = await loadingTask.promise;
          let fullText = "";
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items
              .map((item: any) => item.str)
              .join(" ");
            fullText += pageText + "\n";
          }
          console.log(fullText);
          fullText = fullText.trim();
          if (fullText.length > 0) {
            setManualText(fullText);
            // Extraer nombre si es posible
            const nameMatch = fullText.match(
              /^([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)/m
            );
          } else {
            setManualText("");
            setError(
              "No se pudo extraer texto del PDF. Si es un PDF escaneado, pega el texto manualmente."
            );
          }
        } catch (pdfError) {
          setManualText("");
          console.log(pdfError);
          addLog(
            "Error al procesar el PDF. Si es un PDF escaneado, pega el texto manualmente."
          );
          setError(
            "Error al procesar el PDF. Si es un PDF escaneado, pega el texto manualmente."
          );
        }
      } else {
        throw new Error("Tipo de archivo no soportado. Solo PDF y TXT.");
      }
    } catch (error) {
      addLog(`Error procesando archivo: ${error}`);
      console.error("Error procesando archivo:", error);
      const errorMsg = `Error al procesar archivo: ${
        error instanceof Error ? error.message : "Error desconocido"
      }`;
      setError(errorMsg);
      setManualText(errorMsg);
    } finally {
      setIsExtracting(false);
    }
  };

  const generatePodcastAndAudio = async () => {
    addLog("Iniciando generación de episodio...");
    if (!apiKey.trim()) {
      const errorMsg = "Por favor, ingresa tu API Key de Google AI";
      setError(errorMsg);
      return;
    }

    const textToAnalyze = manualText; // Usar siempre el área manual
    if (!textToAnalyze.trim()) {
      setError("Por favor, proporciona el texto del CV");
      return;
    }

    setIsGeneratingScript(true);
    setError("");
    setPodcastScript("");

    try {
      addLog("Inicializando GoogleGenerativeAI...");
      const genAI = new GoogleGenerativeAI(apiKey);

      // Generate script with Gemini 2.0 Flash
      addLog("Generando episodio con Gemini 2.0 Flash...");
      const model = genAI.getGenerativeModel({
        model: "gemini-2.5-flash-preview-05-20",
      });

      const scriptPrompt = `Bienvenido a The CV Comedy Podcast. Cada CV es un nuevo episodio. Eres un dúo de comediantes profesionales especializados en crear contenido humorístico inteligente para podcasts. Crea el libreto para un episodio de 4-6 minutos que critique de manera divertida y sarcástica un CV (el CV es el invitado del episodio).

FORMATO REQUERIDO para multi-speaker TTS:
${speakers[0].name}: [texto del primer host]
${speakers[1].name}: [texto del segundo host]

CARACTERÍSTICAS DE LOS HOSTS:
- ${speakers[0].name}: ${speakers[0].personality}
- ${speakers[1].name}: ${speakers[1].personality}

ELEMENTOS A CRITICAR CON HUMOR INTELIGENTE:
- Clichés típicos: "soy muy perfeccionista", "trabajo bien en equipo"
- Inconsistencias temporales o lógicas
- Habilidades exageradas: "experto en todo"
- Descripciones pomposas de trabajos básicos
- Objetivos profesionales vagos: "busco crecer profesionalmente"
- Hobbies irrelevantes o clichés
- Errores ortográficos o gramaticales

TONO: Sarcástico pero sofisticado, como un late-night comedy show. Mantén el humor inteligente y evita ser cruel.

IMPORTANTE: 
- Usa EXACTAMENTE "${speakers[0].name}:" y "${speakers[1].name}:" para cada intervención
- Incluye pausas naturales con "[...]"
- Añade énfasis con "[énfasis]" donde sea apropiado
- Haz que la conversación fluya naturalmente
- Máximo 3 minutos y medio. MAXIMO

Analiza este CV y crea el libreto del episodio en español, y bastante bastante crítico (literalmente un roast sin piedad):`;

      addLog("Enviando solicitud para generar episodio...");
      const parts = [`${scriptPrompt}\n\n${textToAnalyze}`];
      const result = await model.generateContent(parts);

      const response = result.response;
      const script = response.text();
      if (!script) throw new Error("No se pudo generar el episodio");
      addLog(`Episodio generado: ${script.substring(0, 100)}...`);
      setPodcastScript(script);
      setIsGeneratingScript(false);

      // Generar audio usando Web Speech API
      setIsGeneratingAudio(true);
      addLog("Generando audio usando Web Speech API...");
      // Usar el episodio generado como prompt para TTS
      const ttsPrompt = script;
      // The Gemini SDK does not support speechConfig in SingleRequestOptions, so we remove it.
      const ttsResult = await new GoogleGenAI({
        apiKey,
      }).models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [
          {
            parts: [
              {
                text:
                  `Genera el audio de un episodio de standup comedy en español, en formato de podcast crítico de CVs, con el tono de un late-night show: sarcástico pero sofisticado, humor inteligente, evita ser cruel. Usa exactamente los nombres de los presentadores para cada intervención, incluye pausas naturales con "[...]" y énfasis con "[énfasis]" donde sea apropiado. Haz que la conversación fluya naturalmente, como un show de comedia nocturno.\n` +
                  ttsPrompt,
              },
            ],
          },
        ],
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            multiSpeakerVoiceConfig: {
              speakerVoiceConfigs: speakers.map((speaker) => ({
                speaker: speaker.name,
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: speaker.voice },
                },
              })),
            },
          },
        },
      });
      const audio = ttsResult.candidates?.[0]?.content?.parts?.[0]?.inlineData;
      if (!audio || !audio.data) throw new Error("No se pudo generar el audio");
      addLog(`Audio generado: ${String(audio.data).substring(0, 100)}...`);
      // Decodifica base64 a Uint8Array (PCM crudo)
      const pcm = Uint8Array.from(atob(String(audio.data)), (c) =>
        c.charCodeAt(0)
      );
      // Convierte PCM a WAV
      const wavBlob = pcmToWav(pcm);
      setAudioBuffer(wavBlob);
    } catch (error) {
      addLog(`Error generando episodio y audio: ${error}`);
      console.error("Error generando episodio y audio:", error);
      setError(
        `Error al generar episodio y audio: ${
          error instanceof Error ? error.message : "Error desconocido"
        }`
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadScript = () => {
    if (podcastScript) {
      const blob = new Blob([podcastScript], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `the-cv-comedy-podcast-episodio-${
        file?.name?.split(".")[0] || "episodio"
      }.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const canGenerate = manualText.trim().length > 0 && apiKey.trim().length > 0;

  useEffect(() => {
    document.addEventListener("DOMContentLoaded", (event) => {
      // @ts-ignore
      paypal
        .HostedButtons({
          hostedButtonId: "6LF6GX88SWCUE",
        })
        .render("#paypal-container-6LF6GX88SWCUE");
    });
  }, []);

  // Feedback visual para procesamiento
  const renderProcessingFeedback = () =>
    isExtracting && (
      <div className="flex items-center justify-center gap-2 mt-4 text-purple-700 text-sm font-semibold">
        <svg
          className="animate-spin h-5 w-5 text-purple-700"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v8z"
          ></path>
        </svg>
        Procesando archivo...
      </div>
    );

  return (
    <>
      <Head>
        <title>
          The CV Comedy Podcast - Convierte cada CV en un episodio humorístico
          con Gemini
        </title>
        <meta
          name="description"
          content="Sube tu CV y genera un episodio humorístico de The CV Comedy Podcast usando Google Gemini 2.0 Flash con multi-speaker TTS"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta
          name="keywords"
          content="podcast, CV, comedia, Google Gemini, TTS, inteligencia artificial, humor"
        />
        <meta
          property="og:title"
          content="The CV Comedy Podcast - Convierte CVs en episodios humorísticos"
        />
        <meta
          property="og:description"
          content="Convierte tu CV en un episodio humorístico con Google Gemini 2.0 Flash y Multi-Speaker TTS."
        />
        <meta property="og:image" content="/cover.png" />
        <meta
          property="og:url"
          content="https://the-cv-comedy-podcast.vercel.app/"
        />
        <meta property="og:type" content="website" />

        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="The CV Comedy Podcast" />
        <meta
          name="twitter:description"
          content="Convierte tu CV en un episodio humorístico con Google Gemini 2.0 Flash y Multi-Speaker TTS."
        />
        <meta name="twitter:image" content="/cover.png" />
      </Head>
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mt-8 mb-2">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              🎙️ The CV Comedy Podcast
            </h1>
            <p className="text-lg text-gray-600 mb-4">
              Convierte cada CV en un episodio humorístico usando Google Gemini
              2.0 Flash
            </p>
            <div className="flex justify-center items-center gap-2 text-sm text-gray-500">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full">
                ✨ Multi-Speaker TTS
              </span>
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                🤖 Gemini 2.0 Flash
              </span>
              <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full">
                😄 Humor Inteligente
              </span>
            </div>
          </div>

          {/* API Key Section */}
          <div className="p-4">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              🔑 Configuración
            </h2>
            <div className="mb-4">
              <label
                htmlFor="apiKey"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Google AI API Key
              </label>
              <input
                type="password"
                id="apiKey"
                value={apiKey}
                onChange={(e) => (
                  setApiKey(e.target.value),
                  localStorage.setItem("apiKey", e.target.value)
                )}
                placeholder="Ingresa tu API Key de Google AI Studio"
                className="bg-white w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Obtén tu API Key gratis en{" "}
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:underline"
                >
                  Google AI Studio
                </a>
              </p>
            </div>
          </div>

          {/* Upload Section + Área manual con toggle */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-2">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Sube tu CV
            </h2>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 hover:border-gray-400"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{ cursor: "pointer" }}
            >
              <div className="mb-4">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              {file ? (
                <>
                  <p className="text-green-600 font-medium">✅ {file.name}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    Archivo cargado correctamente
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </>
              ) : (
                <>
                  <p className="text-lg text-gray-600 mb-2">
                    Arrastra tu CV aquí o haz clic para seleccionar
                  </p>
                  <p className="text-sm text-gray-500">
                    Soporta archivos PDF y TXT
                  </p>
                </>
              )}
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".pdf,.txt"
                onChange={(e) => handleFileUpload(e.target.files![0])}
              />
            </div>
            {/* Feedback visual mientras se procesa el archivo */}
            {renderProcessingFeedback()}
            {/* Toggle para mostrar área manual solo si no hay archivo */}
            {!file && (
              <div className="mt-4 text-center">
                <button
                  onClick={() => setShowManualInput((v) => !v)}
                  className="text-blue-500 hover:text-blue-700 text-sm underline"
                  disabled={isExtracting}
                >
                  {showManualInput
                    ? "Ocultar entrada manual"
                    : "Añadir texto manualmente"}
                </button>
              </div>
            )}
            {/* Área manual: visible si hay archivo o si el toggle está abierto */}
            {(file || showManualInput) && (
              <textarea
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                placeholder={
                  file
                    ? "Aquí aparecerá el texto extraído de tu CV. Puedes editarlo antes de generar el episodio..."
                    : "Pega aquí el texto de tu CV manualmente..."
                }
                className="w-full mt-4 h-40 p-4 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isExtracting}
              />
            )}
          </div>

          {/* Generate Button */}
          {canGenerate && !podcastScript && !devMode && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                Generar episodio y audio
              </h2>
              <button
                onClick={generatePodcastAndAudio}
                disabled={isGenerating || isExtracting}
                className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors w-full flex items-center justify-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    Generando...
                  </>
                ) : (
                  "🎭 Generar episodio y audio"
                )}
              </button>
              {/* Feedback visual debajo del botón */}
              <div className="mt-4 min-h-[32px] flex flex-col items-center">
                {error && (
                  <div className="mt-2 text-red-600 text-sm font-semibold text-center">
                    {error}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Podcast Script and Audio */}
          {(podcastScript || devMode) && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6 flex flex-col md:flex-row gap-6">
              {/* Script a la izquierda */}
              <div className="flex-1 basis-1/2 min-w-0">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Episodio generado
                </h2>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-h-96 overflow-y-auto whitespace-pre-wrap text-gray-700">
                  {devMode
                    ? "[Episodio de prueba]\nAlex: Bienvenidos a The CV Comedy Podcast.\nSam: Hoy solo estamos probando el visualizador de audio.\n[...]"
                    : podcastScript}
                </div>
                {podcastScript && !devMode && (
                  <button
                    onClick={downloadScript}
                    className="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors w-full"
                  >
                    📥 Descargar episodio
                  </button>
                )}
              </div>
              {/* Portada + audio/ProgressBar a la derecha */}
              <div className="flex-1 basis-1/2 min-w-0 flex flex-col items-center">
                <div className="text-2xl font-semibold text-gray-800 mb-4 md:h-lh">
                  <h2 className="md:hidden">Audio generado</h2>
                </div>
                {devMode ? (
                  audioBuffer ? (
                    <div className="w-full flex flex-col items-center">
                      <AudioVisualizer audioBuffer={audioBuffer} />
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center min-h-[80px] w-full">
                      {/* Portada y canvas aunque no haya audio */}
                      <div className="relative w-full max-w-md aspect-square rounded-lg overflow-hidden shadow-lg">
                        <img
                          src="/cover.png"
                          alt="Portada del episodio"
                          className="absolute inset-0 w-full h-full object-cover z-0"
                        />
                        <canvas
                          width={800}
                          height={800}
                          className="absolute inset-0 w-full h-full z-10 pointer-events-none opacity-75"
                        />
                      </div>
                      {/* ProgressBar mientras se genera el audio */}
                      <div className="w-full max-w-md mt-2">
                        <ProgressBar duration={120} />
                      </div>
                      <p className="text-gray-500 mt-2 text-center">
                        Generando audio del episodio... Esto puede tardar unos
                        minutos.
                      </p>
                    </div>
                  )
                ) : audioBuffer ? (
                  <div className="w-full flex flex-col items-center">
                    <AudioVisualizer audioBuffer={audioBuffer} />
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center min-h-[80px] w-full">
                    {/* Portada y canvas aunque no haya audio */}
                    <div className="relative w-full max-w-md aspect-square rounded-lg overflow-hidden shadow-lg">
                      <img
                        src="/cover.png"
                        alt="Portada del episodio"
                        className="absolute inset-0 w-full h-full object-cover z-0"
                      />
                      <canvas
                        width={800}
                        height={800}
                        className="absolute inset-0 w-full h-full z-10 pointer-events-none opacity-75"
                      />
                    </div>
                    {/* ProgressBar mientras se genera el audio */}
                    <div className="w-full max-w-md mt-2">
                      <ProgressBar duration={120} />
                    </div>
                    <p className="text-gray-500 mt-2">
                      Generando audio del episodio... Esto puede tardar unos
                      minutos.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Donation Button at bottom */}
          {podcastScript && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6 flex flex-col items-center">
              <h3 className="text-lg font-semibold text-gray-800 mb-2 text-center">
                ❤️ Apoya The CV Comedy Podcast
              </h3>
              <p className="text-gray-600 mb-4 text-sm max-w-md mx-auto text-center">
                Si te gusta este proyecto, considera hacer una donación para
                mantener nuevos episodios saliendo al aire
              </p>
              {/* PayPal Donation Link */}
              <a
                href="https://www.paypal.com/ncp/payment/6LF6GX88SWCUE"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 bg-blue-400 hover:bg-blue-500 text-gray-900 font-semibold px-6 py-2 rounded-lg shadow transition-colors text-lg"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="7.056000232696533 3 37.35095977783203 45"
                  className="h-6 w-6"
                  fill="none"
                >
                  <g clipPath="url(#a)">
                    <path
                      fill="#002991"
                      d="M38.914 13.35c0 5.574-5.144 12.15-12.927 12.15H18.49l-.368 2.322L16.373 39H7.056l5.605-36h15.095c5.083 0 9.082 2.833 10.555 6.77a9.687 9.687 0 0 1 .603 3.58z"
                    />
                    <path
                      fill="#60CDFF"
                      d="M44.284 23.7A12.894 12.894 0 0 1 31.53 34.5h-5.206L24.157 48H14.89l1.483-9 1.75-11.178.367-2.322h7.497c7.773 0 12.927-6.576 12.927-12.15 3.825 1.974 6.055 5.963 5.37 10.35z"
                    />
                    <path
                      fill="#008CFF"
                      d="M38.914 13.35C37.31 12.511 35.365 12 33.248 12h-12.64L18.49 25.5h7.497c7.773 0 12.927-6.576 12.927-12.15z"
                    />
                  </g>
                  <defs>
                    <clipPath id="a">
                      <rect
                        x="7.056"
                        y="3"
                        width="37.351"
                        height="45"
                        fill="white"
                      />
                    </clipPath>
                  </defs>
                </svg>
                Donar con PayPal
              </a>
            </div>
          )}

          {/* Debug Log solo si dev=0 o dev=1 en el search param */}
          {debugLog.length > 0 &&
            (devMode ||
              (typeof window !== "undefined" &&
                window.location.search.includes("dev=0"))) && (
              <div className="bg-gray-100 rounded-lg p-4 mb-6 max-h-48 overflow-y-auto font-mono text-xs text-gray-600">
                <h3 className="font-semibold mb-2">Logs de depuración:</h3>
                {debugLog.map((log, idx) => (
                  <div key={idx}>{log}</div>
                ))}
              </div>
            )}

          {/* Footer */}
          <div className="text-center mt-8 text-gray-500 text-sm flex flex-col items-center gap-1">
            <p>
              ⚠️ Aplicación desarrollada con Vibe Coding (AI) Usa API de Google
              AI directamente desde el navegador.
            </p>
            <a
              href="https://github.com/nivandres/the-cv-comedy-podcast"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 mt-2 px-4 py-2 bg-gray-900 text-white rounded-lg shadow hover:bg-gray-800 transition-colors text-base font-semibold"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.729.083-.729 1.205.085 1.84 1.237 1.84 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.775.418-1.305.762-1.606-2.665-.304-5.466-1.334-5.466-5.931 0-1.31.468-2.381 1.236-3.221-.124-.303-.535-1.523.117-3.176 0 0 1.008-.322 3.3 1.23.957-.266 1.984-.399 3.003-.404 1.018.005 2.046.138 3.006.404 2.289-1.552 3.295-1.23 3.295-1.23.653 1.653.242 2.873.119 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.804 5.625-5.475 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 0 .322.218.694.825.576C20.565 21.796 24 17.299 24 12c0-6.627-5.373-12-12-12z" />
              </svg>
              Repositorio en GitHub
            </a>
          </div>
        </div>
      </div>
    </>
  );
}
