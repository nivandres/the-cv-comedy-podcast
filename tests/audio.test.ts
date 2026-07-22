import { describe, it, expect } from "vitest";
import {
  pcmToWav,
  wavDuration,
  concatPcm,
  base64ToUint8Array,
  arrayBufferToBase64,
  parseSampleRate,
} from "@/lib/audio";

describe("pcmToWav", () => {
  it("produce un WAV válido: cabecera de 44 bytes + PCM", async () => {
    const pcm = new Uint8Array([1, 2, 3, 4]);
    const blob = pcmToWav(pcm, 24000);
    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + pcm.length);

    const bytes = new Uint8Array(await blob.arrayBuffer());
    const ascii = (from: number, to: number) =>
      String.fromCharCode(...bytes.slice(from, to));
    expect(ascii(0, 4)).toBe("RIFF");
    expect(ascii(8, 12)).toBe("WAVE");
    expect(ascii(36, 40)).toBe("data");

    const view = new DataView(bytes.buffer);
    expect(view.getUint32(24, true)).toBe(24000); // sample rate
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint16(34, true)).toBe(16); // bits por muestra
    expect(view.getUint32(40, true)).toBe(pcm.length); // tamaño de datos
    expect(bytes.slice(44)).toEqual(pcm);
  });
});

describe("wavDuration", () => {
  it("calcula la duración de un WAV PCM 16-bit mono", () => {
    // 2 segundos a 24 kHz → 24000 * 2 bytes * 2 s de PCM
    const pcm = new Uint8Array(24000 * 2 * 2);
    const blob = pcmToWav(pcm, 24000);
    expect(wavDuration(blob, 24000)).toBeCloseTo(2, 5);
  });

  it("nunca devuelve negativo", () => {
    expect(wavDuration(new Blob([new Uint8Array(10)]), 24000)).toBe(0);
  });
});

describe("concatPcm", () => {
  it("concatena chunks en orden", () => {
    const out = concatPcm([
      new Uint8Array([1, 2]),
      new Uint8Array([]),
      new Uint8Array([3]),
    ]);
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it("lista vacía produce array vacío", () => {
    expect(concatPcm([]).length).toBe(0);
  });
});

describe("base64", () => {
  it("round-trip de bytes arbitrarios", () => {
    const original = new Uint8Array(70000).map((_, i) => i % 256);
    const base64 = arrayBufferToBase64(original.buffer);
    const decoded = base64ToUint8Array(base64);
    expect(decoded).toEqual(original);
  });
});

describe("parseSampleRate", () => {
  it("extrae el rate del mimeType de Gemini TTS", () => {
    expect(parseSampleRate("audio/L16;codec=pcm;rate=24000")).toBe(24000);
    expect(parseSampleRate("audio/L16;rate=44100")).toBe(44100);
  });

  it("usa 24000 por defecto", () => {
    expect(parseSampleRate(undefined)).toBe(24000);
    expect(parseSampleRate("audio/wav")).toBe(24000);
  });
});
