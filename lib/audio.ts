// Utilidades de audio para The CV Comedy Podcast: PCM/WAV, base64 y
// exportación del episodio como video (portada + visualizador + audio).

// Gradiente de marca (portada): compartido por la UI, el visualizador y el video
export const BRAND_STOPS = ["#f72585", "#b5179e", "#ff8800"] as const;

// Sample rate que emiten los modelos TTS de Gemini (PCM 16-bit mono)
export const DEFAULT_TTS_SAMPLE_RATE = 24000;

// Duración en segundos de un WAV PCM 16-bit mono generado por pcmToWav
export function wavDuration(blob: Blob, sampleRate: number): number {
  return Math.max(0, (blob.size - 44) / (sampleRate * 2));
}

// Dibuja un frame del visualizador de espectro (barras con el gradiente de
// marca). Compartido entre el reproductor y la exportación de video.
export function drawSpectrumFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dataArray: Uint8Array
) {
  const barWidth = (width / dataArray.length) * 2.5;
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, BRAND_STOPS[0]);
  grad.addColorStop(0.5, BRAND_STOPS[1]);
  grad.addColorStop(1, BRAND_STOPS[2]);
  let x = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const barHeight = dataArray[i] * 0.7;
    ctx.fillStyle = grad;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    ctx.globalAlpha = 1;
    x += barWidth + 1;
  }
}

// Envuelve PCM crudo (salida de Gemini TTS) en una cabecera WAV
export function pcmToWav(
  pcm: Uint8Array,
  sampleRate = DEFAULT_TTS_SAMPLE_RATE,
  numChannels = 1,
  bitsPerSample = 16
) {
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const wavBuffer = new ArrayBuffer(44 + pcm.length);
  const view = new DataView(wavBuffer);

  view.setUint32(0, 0x52494646, false); // 'RIFF'
  view.setUint32(4, 36 + pcm.length, true);
  view.setUint32(8, 0x57415645, false); // 'WAVE'
  view.setUint32(12, 0x666d7420, false); // 'fmt '
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  view.setUint32(36, 0x64617461, false); // 'data'
  view.setUint32(40, pcm.length, true);
  new Uint8Array(wavBuffer, 44).set(pcm);

  return new Blob([wavBuffer], { type: "audio/wav" });
}

export function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

// El mimeType del audio viene como "audio/L16;codec=pcm;rate=24000"
export function parseSampleRate(mimeType?: string): number {
  const match = mimeType?.match(/rate=(\d+)/);
  return match ? parseInt(match[1], 10) : DEFAULT_TTS_SAMPLE_RATE;
}

export function concatPcm(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Exporta el episodio como video: portada + barras del visualizador + audio.
// Se graba en tiempo real con MediaRecorder (dura lo mismo que el episodio).
export async function exportEpisodeVideo(
  audioBlob: Blob,
  coverSrc: string,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal
): Promise<Blob> {
  const audioCtx = new AudioContext();
  try {
    // Por si el contexto arranca suspendido (política de autoplay)
    await audioCtx.resume();
    const audioData = await audioBlob.arrayBuffer();
    const buffer = await audioCtx.decodeAudioData(audioData);

    const canvas = document.createElement("canvas");
    canvas.width = 720;
    canvas.height = 720;
    const ctx = canvas.getContext("2d")!;
    const cover = new Image();
    cover.src = coverSrc;
    await cover.decode();

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64;
    const dest = audioCtx.createMediaStreamDestination();
    // Solo al destino de grabación: la exportación es silenciosa
    source.connect(analyser);
    analyser.connect(dest);

    const stream = canvas.captureStream(30);
    dest.stream.getAudioTracks().forEach((track) => stream.addTrack(track));
    const mimeType = MediaRecorder.isTypeSupported("video/mp4")
      ? "video/mp4"
      : "video/webm";
    const recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 2_500_000,
    });
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    let animationId = 0;
    const startTime = audioCtx.currentTime;

    const draw = () => {
      analyser.getByteFrequencyData(dataArray);
      ctx.drawImage(cover, 0, 0, canvas.width, canvas.height);
      drawSpectrumFrame(ctx, canvas.width, canvas.height, dataArray);
      onProgress(
        Math.min((audioCtx.currentTime - startTime) / buffer.duration, 1)
      );
      animationId = requestAnimationFrame(draw);
    };

    return await new Promise<Blob>((resolve, reject) => {
      recorder.onstop = () => {
        cancelAnimationFrame(animationId);
        if (signal?.aborted) reject(new DOMException("Aborted", "AbortError"));
        else resolve(new Blob(chunks, { type: mimeType }));
      };
      recorder.onerror = () => {
        cancelAnimationFrame(animationId);
        reject(new Error("Error grabando el video"));
      };
      source.onended = () => {
        // Pequeño margen para no cortar el final del audio
        setTimeout(() => recorder.state !== "inactive" && recorder.stop(), 300);
      };
      // Si cancelaron durante el setup (decodificación/carga de portada),
      // no llegar a grabar la duración completa del episodio
      if (signal?.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      signal?.addEventListener("abort", () => {
        source.stop();
        if (recorder.state !== "inactive") recorder.stop();
      });
      recorder.start(1000);
      source.start();
      draw();
    });
  } finally {
    audioCtx.close();
  }
}
