// Reproductor único del episodio: portada + visualizador + <audio> persistente.
// Reproduce los segmentos según van llegando y, cuando el episodio completo
// está listo, cambia al audio completo restaurando la posición de escucha
// (mismo elemento <audio>: sin remount, sin saltos de layout).
import { useState, useRef, useEffect, useMemo } from "react";
import Image from "next/image";
import { drawSpectrumFrame, wavDuration } from "@/lib/audio";
import { parseTurns, pickTurn, type SubtitleTurn } from "@/lib/subtitles";
import { useTranslation } from "@/i18n/translation";
import { Spinner } from "@/components/ui";

export function EpisodePlayer({
  segments,
  fullAudio,
  totalSegments,
  segmentTexts,
  speakerNames,
  isGenerating,
  sampleRate,
}: {
  segments: Blob[];
  fullAudio: Blob | null;
  totalSegments: number;
  segmentTexts: string[];
  speakerNames: string[];
  isGenerating: boolean;
  sampleRate: number;
}) {
  const t = useTranslation();
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [segIndex, setSegIndex] = useState(0);
  const [waiting, setWaiting] = useState(false);
  const [playBlocked, setPlayBlocked] = useState(false);
  const [url, setUrl] = useState("");

  // Al pasar a audio completo: posición global = duración de los segmentos
  // anteriores + posición actual; se aplica cuando cargue el nuevo src.
  const pendingSeekRef = useRef<{ time: number; play: boolean } | null>(null);
  const autoPlayNextRef = useRef(false);
  const wasFullRef = useRef(false);

  const currentBlob = fullAudio ?? segments[segIndex] ?? null;

  // Captura la posición ANTES de que el src cambie al episodio completo
  // (este efecto corre antes que el del object URL, con el src viejo aún activo)
  useEffect(() => {
    if (fullAudio && !wasFullRef.current) {
      wasFullRef.current = true;
      const audio = audioRef.current;
      if (audio && segments.length > 0) {
        let offset = 0;
        for (let i = 0; i < Math.min(segIndex, segments.length); i++) {
          offset += wavDuration(segments[i], sampleRate);
        }
        pendingSeekRef.current = {
          time: offset + audio.currentTime,
          play: !audio.paused && !audio.ended,
        };
      }
    } else if (!fullAudio && wasFullRef.current) {
      // Regeneración: volver al modo por segmentos desde el principio
      wasFullRef.current = false;
      pendingSeekRef.current = null;

      setSegIndex(0);
      setWaiting(false);
    }
  }, [fullAudio, segIndex, segments, sampleRate]);

  // Object URL del blob actual: crear/revocar con ciclo de vida del efecto
  useEffect(() => {
    if (!currentBlob) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- sincroniza recurso externo (object URL)
      setUrl("");
      return;
    }
    const nextUrl = URL.createObjectURL(currentBlob);

    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [currentBlob]);

  // Si estábamos esperando la siguiente parte y ya llegó, continuar solos.
  // También cubre el caso de reanudar tras un error: el <audio> quedó en
  // "ended" sin que se marcara waiting, y llegan partes nuevas.
  useEffect(() => {
    const stuckAtEnd = Boolean(audioRef.current?.ended) && !fullAudio;
    if ((waiting || stuckAtEnd) && segIndex + 1 < segments.length) {
      setWaiting(false);
      setSegIndex((i) => i + 1);
      autoPlayNextRef.current = true;
    }
  }, [segments.length, waiting, segIndex, fullAudio]);

  const handleLoadedMetadata = () => {
    const audio = audioRef.current;
    if (!audio) return;
    const seek = pendingSeekRef.current;
    if (seek) {
      pendingSeekRef.current = null;
      audio.currentTime = Math.min(seek.time, audio.duration || seek.time);
      // Continuar si se estaba reproduciendo O si había un autoplay
      // encadenado pendiente (p. ej. esperando la última parte cuando
      // llegó junto con el episodio completo)
      if (seek.play || autoPlayNextRef.current) {
        autoPlayNextRef.current = false;
        audio.play().catch(() => setPlayBlocked(true));
      }
      return;
    }
    if (autoPlayNextRef.current) {
      autoPlayNextRef.current = false;
      audio.play().catch(() => setPlayBlocked(true));
    }
  };

  const handleEnded = () => {
    if (fullAudio) return; // fin natural del episodio completo
    if (segIndex + 1 < segments.length) {
      setSegIndex(segIndex + 1);
      autoPlayNextRef.current = true;
    } else if (isGenerating) {
      setWaiting(true);
    }
  };

  // ── Subtítulos: aproximación por peso de texto (el TTS no da timestamps).
  // Desactivados por defecto: la sincronización es aproximada, que cada
  // usuario decida activarlos con el botón CC. ──
  const [subtitlesOn, setSubtitlesOn] = useState(false);
  const [subtitle, setSubtitle] = useState<SubtitleTurn | null>(null);
  const parsedSegments = useMemo(
    () => segmentTexts.map((text) => parseTurns(text, speakerNames)),
    [segmentTexts, speakerNames]
  );

  // force=true recalcula también en pausa (seek manual, re-activar CC)
  const updateSubtitle = (force = false) => {
    const audio = audioRef.current;
    if (!audio || parsedSegments.length === 0) return;
    if (!force && audio.paused && !audio.ended) return;

    // Localiza el segmento activo y la fracción transcurrida dentro de él
    let activeSegment = segIndex;
    let fraction = audio.duration > 0 ? audio.currentTime / audio.duration : 0;
    if (fullAudio) {
      let t = audio.currentTime;
      activeSegment = 0;
      for (let i = 0; i < segments.length; i++) {
        const dur = wavDuration(segments[i], sampleRate);
        if (t <= dur || i === segments.length - 1) {
          activeSegment = i;
          fraction = dur > 0 ? Math.min(t / dur, 1) : 0;
          break;
        }
        t -= dur;
      }
    }

    const current = pickTurn(parsedSegments[activeSegment] ?? [], fraction);
    if (current && current !== subtitle) setSubtitle(current);
  };

  const handleTimeUpdate = () => {
    if (subtitlesOn) updateSubtitle();
  };

  // Visualizador: analiza el MISMO <audio>; el grafo se crea una sola vez
  // en el primer play (gesto del usuario) y el rAF se detiene al pausar.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const audio = audioRef.current;
    const canvas = canvasRef.current;
    if (!audio || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let dataArray = new Uint8Array(0);

    const draw = () => {
      const analyser = analyserRef.current;
      if (!analyser) return;
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawSpectrumFrame(ctx, canvas.width, canvas.height, dataArray);
      rafRef.current = requestAnimationFrame(draw);
    };
    // Respeta la preferencia de movimiento reducido: sin animación de barras
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    const handlePlay = () => {
      setPlayBlocked(false);
      if (!audioCtxRef.current) {
        const audioCtx = new (
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext })
            .webkitAudioContext
        )();
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        const source = audioCtx.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
        audioCtxRef.current = audioCtx;
        analyserRef.current = analyser;
      }
      audioCtxRef.current.resume();
      if (reducedMotion) return;
      dataArray = new Uint8Array(analyserRef.current!.frequencyBinCount);
      cancelAnimationFrame(rafRef.current);
      draw();
    };
    const stopDrawing = () => cancelAnimationFrame(rafRef.current);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", stopDrawing);
    audio.addEventListener("ended", stopDrawing);
    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", stopDrawing);
      audio.removeEventListener("ended", stopDrawing);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      audioCtxRef.current?.close();
      audioCtxRef.current = null;
      analyserRef.current = null;
    };
  }, []);

  const hasAudio = Boolean(currentBlob);
  const partVariables = { current: segIndex + 1, total: totalSegments };
  const statusText = fullAudio
    ? t.player.complete
    : playBlocked
      ? t.player.playBlocked(partVariables)
      : waiting
        ? t.player.waiting
        : hasAudio
          ? t.player.part(partVariables)
          : isGenerating
            ? t.player.preparing
            : t.player.empty;

  return (
    <div className="flex w-full flex-col items-center gap-2">
      <div className="relative aspect-square w-full max-w-md overflow-hidden rounded-2xl shadow-lg">
        <Image
          src="/cover.png"
          alt={String(t.player.cover)}
          fill
          sizes="(max-width: 768px) 100vw, 448px"
          className={`absolute inset-0 z-0 object-cover transition-opacity ${
            hasAudio ? "opacity-100" : "opacity-60"
          }`}
        />
        <canvas
          ref={canvasRef}
          width={800}
          height={800}
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-10 h-full w-full opacity-75"
        />
        {!hasAudio && (
          <div className="absolute inset-0 z-20 flex items-center justify-center">
            {isGenerating && <Spinner className="h-8 w-8 text-white" />}
          </div>
        )}
        {/* Subtítulos sobre la portada */}
        {subtitlesOn && subtitle && hasAudio && (
          <div className="absolute bottom-2 left-2 right-2 z-20 rounded-lg bg-black/60 px-3 py-2 text-center text-sm text-white backdrop-blur-sm">
            {subtitle.speaker && (
              <span className="font-semibold">{subtitle.speaker}: </span>
            )}
            {subtitle.text}
          </div>
        )}
      </div>
      <audio
        ref={audioRef}
        src={url || undefined}
        controls
        className="w-full max-w-md"
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        onTimeUpdate={handleTimeUpdate}
        onSeeked={() => subtitlesOn && updateSubtitle(true)}
        aria-label={String(t.player.audioLabel)}
      />
      <div className="flex w-full max-w-md items-center justify-between gap-2">
        <p
          aria-live="polite"
          className="min-w-0 flex-1 text-center text-xs text-zinc-500 dark:text-zinc-400"
        >
          {statusText}
        </p>
        {hasAudio && (
          <button
            onClick={() => {
              const next = !subtitlesOn;
              setSubtitlesOn(next);
              setSubtitle(null);
              if (next) updateSubtitle(true);
            }}
            aria-pressed={subtitlesOn}
            className={`shrink-0 rounded-full px-2 py-0.5 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-purple-500 ${
              subtitlesOn
                ? "bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-200"
                : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400"
            }`}
          >
            {t.player.cc}
          </button>
        )}
      </div>
    </div>
  );
}
