import { describe, it, expect, vi } from "vitest";
import {
  splitScriptIntoSegments,
  speakerLineRegex,
  callWithFallback,
  describeApiError,
  isTransientApiError,
} from "@/lib/gemini";

const SPEAKERS = ["Alex", "Sam"];

function makeScript(turnCount: number, turnLength = 80): string {
  return Array.from({ length: turnCount }, (_, i) => {
    const speaker = i % 2 === 0 ? "Alex" : "Sam";
    return `${speaker}: ${"palabra ".repeat(Math.ceil(turnLength / 8))}`.trim();
  }).join("\n");
}

describe("speakerLineRegex", () => {
  it("con nombres restringe a los speakers reales", () => {
    const regex = speakerLineRegex(SPEAKERS);
    expect(regex.test("Alex: hola")).toBe(true);
    expect(regex.test("  Sam : con espacios")).toBe(true);
    expect(regex.test("Nota: risas del público")).toBe(false);
    expect(regex.test("12:30 de la tarde")).toBe(false);
  });

  it("sin nombres usa el patrón genérico", () => {
    const regex = speakerLineRegex();
    expect(regex.test("Cualquiera: texto")).toBe(true);
  });

  it("escapa caracteres especiales de los nombres", () => {
    const regex = speakerLineRegex(["Dr. X"]);
    expect(regex.test("Dr. X: hola")).toBe(true);
    expect(regex.test("DrZX: hola")).toBe(false);
  });
});

describe("splitScriptIntoSegments", () => {
  it("un guion corto queda en un solo segmento", () => {
    const script = "Alex: hola\nSam: adiós";
    expect(splitScriptIntoSegments(script, 1800, SPEAKERS)).toEqual([script]);
  });

  it("divide guiones largos sin perder contenido", () => {
    const script = makeScript(40);
    const segments = splitScriptIntoSegments(script, 1800, SPEAKERS);
    expect(segments.length).toBeGreaterThan(1);
    expect(segments.join("\n")).toBe(script);
  });

  it("cada segmento empieza en un turno de speaker", () => {
    const segments = splitScriptIntoSegments(makeScript(40), 1800, SPEAKERS);
    for (const segment of segments) {
      expect(segment).toMatch(/^(Alex|Sam):/);
    }
  });

  it("respeta aproximadamente el máximo de caracteres", () => {
    const segments = splitScriptIntoSegments(makeScript(60), 500, SPEAKERS);
    for (const segment of segments.slice(0, -1)) {
      // un turno entero puede exceder el límite, pero no por más de un turno
      expect(segment.length).toBeLessThan(500 + 200);
    }
  });

  it("no corta en líneas que no son speakers reales", () => {
    const script =
      "Alex: " +
      "a".repeat(900) +
      "\nNota: esto no es un turno\nSam: " +
      "b".repeat(900);
    const segments = splitScriptIntoSegments(script, 1000, SPEAKERS);
    // "Nota:" queda pegada al turno de Alex, nunca abre segmento
    expect(segments.every((s) => !s.startsWith("Nota:"))).toBe(true);
  });

  it("texto sin speakers devuelve el guion entero", () => {
    expect(splitScriptIntoSegments("solo un párrafo", 1800, SPEAKERS)).toEqual([
      "solo un párrafo",
    ]);
  });
});

describe("isTransientApiError / describeApiError", () => {
  it("clasifica 503 y 429 como transitorios", () => {
    expect(
      isTransientApiError(new Error('{"code":503,"status":"UNAVAILABLE"}'))
    ).toBe(true);
    expect(isTransientApiError(new Error("high demand"))).toBe(true);
    expect(isTransientApiError(new Error("RESOURCE_EXHAUSTED"))).toBe(true);
    expect(isTransientApiError(new Error("API key not valid"))).toBe(false);
  });

  it("traduce errores a mensajes accionables", () => {
    expect(
      describeApiError(new Error("503 UNAVAILABLE high demand"))
    ).toContain("saturados");
    expect(describeApiError(new Error("429 RESOURCE_EXHAUSTED"))).toContain(
      "cuota"
    );
    expect(describeApiError(new Error("API key not valid"))).toContain(
      "API Key"
    );
    expect(describeApiError(new Error("otra cosa"))).toBe("otra cosa");
    expect(describeApiError("")).toBe("Error desconocido");
  });
});

describe("callWithFallback", () => {
  it("devuelve el primer intento exitoso con su índice", async () => {
    const { result, modelIdx } = await callWithFallback(
      ["a", "b"],
      async (m) => `ok-${m}`,
      () => {}
    );
    expect(result).toBe("ok-a");
    expect(modelIdx).toBe(0);
  });

  it("reintenta errores transitorios con espera y luego recupera", async () => {
    vi.useFakeTimers();
    let calls = 0;
    const promise = callWithFallback(
      ["a"],
      async () => {
        calls++;
        if (calls < 2) throw new Error("503 UNAVAILABLE");
        return "ok";
      },
      () => {}
    );
    await vi.advanceTimersByTimeAsync(15_000);
    const { result } = await promise;
    expect(result).toBe("ok");
    expect(calls).toBe(2);
    vi.useRealTimers();
  });

  it("errores no transitorios saltan al siguiente modelo sin reintentar", async () => {
    let callsA = 0;
    const { result, modelIdx } = await callWithFallback(
      ["a", "b"],
      async (m) => {
        if (m === "a") {
          callsA++;
          throw new Error("API key not valid");
        }
        return "ok-b";
      },
      () => {}
    );
    expect(callsA).toBe(1);
    expect(result).toBe("ok-b");
    expect(modelIdx).toBe(1);
  });

  it("startIdx arranca en el modelo indicado (índice pegajoso)", async () => {
    const attempted: string[] = [];
    const { result } = await callWithFallback(
      ["a", "b"],
      async (m) => {
        attempted.push(m);
        return `ok-${m}`;
      },
      () => {},
      1
    );
    expect(attempted).toEqual(["b"]);
    expect(result).toBe("ok-b");
  });

  it("si todos fallan lanza el último error", async () => {
    await expect(
      callWithFallback(
        ["a", "b"],
        async (m) => {
          throw new Error(`fallo-${m}`);
        },
        () => {}
      )
    ).rejects.toThrow("fallo-b");
  });
});
