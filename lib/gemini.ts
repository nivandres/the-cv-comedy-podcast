// Lógica de acceso a Gemini: modelos, reintentos con fallback y utilidades
// compartidas entre la generación de libreto, OCR y TTS.
import type { GoogleGenAI } from "@google/genai";

// Tamaño máximo (en caracteres) de cada segmento de TTS
export const TTS_SEGMENT_MAX_CHARS = 1800;

// Carga el SDK bajo demanda: solo se necesita tras la primera interacción,
// así no viaja en el bundle inicial de la página.
export async function createClient(apiKey: string): Promise<GoogleGenAI> {
  const { GoogleGenAI } = await import("@google/genai");
  return new GoogleGenAI({ apiKey });
}

// Modelos en orden de preferencia: si uno falla o está saturado (503),
// se reintenta con espera y luego se pasa al siguiente automáticamente.
export const SCRIPT_MODELS = [
  "gemini-3.5-flash",
  "gemini-2.5-flash",
  "gemini-flash-lite-latest",
];

export const TTS_MODELS = [
  "gemini-3.1-flash-tts-preview",
  "gemini-2.5-pro-preview-tts",
  "gemini-2.5-flash-preview-tts",
];

// Errores transitorios de la API: saturación del modelo (503) o cuota (429)
export function isTransientApiError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return /429|RESOURCE_EXHAUSTED|503|UNAVAILABLE|overloaded|high demand/i.test(
    message
  );
}

// Traduce los errores crudos de la API a mensajes accionables para el usuario
export function describeApiError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  if (/503|UNAVAILABLE|overloaded|high demand/i.test(message)) {
    return "Los modelos de Gemini están saturados en este momento (error 503). Suele ser temporal: espera un par de minutos y reintenta.";
  }
  if (/429|RESOURCE_EXHAUSTED|quota/i.test(message)) {
    return "Alcanzaste el límite de cuota de tu API Key (error 429). Espera un minuto y reintenta, o revisa tu plan en Google AI Studio.";
  }
  if (
    /API key not valid|API_KEY_INVALID|PERMISSION_DENIED|401|403/i.test(message)
  ) {
    return "La API Key no es válida o no tiene permisos. Revísala en Google AI Studio.";
  }
  return message || "Error desconocido";
}

// Ejecuta una llamada a la API probando modelos en orden: los errores
// transitorios (429/503) se reintentan con espera creciente antes de pasar
// al siguiente modelo. Devuelve también el índice del modelo que funcionó
// para poder mantenerlo en llamadas posteriores (voz/estilo consistente).
export async function callWithFallback<T>(
  models: string[],
  attempt: (model: string) => Promise<T>,
  log: (message: string) => void,
  startIdx = 0
): Promise<{ result: T; modelIdx: number }> {
  let lastError: unknown = null;
  for (let idx = Math.max(0, startIdx); idx < models.length; idx++) {
    const model = models[idx];
    for (let attemptNum = 1; attemptNum <= 3; attemptNum++) {
      try {
        return { result: await attempt(model), modelIdx: idx };
      } catch (err) {
        lastError = err;
        if (isTransientApiError(err) && attemptNum < 3) {
          const waitSeconds = attemptNum * 15;
          log(
            `${model} saturado o sin cuota (intento ${attemptNum}/3), esperando ${waitSeconds}s...`
          );
          await new Promise((r) => setTimeout(r, waitSeconds * 1000));
          continue;
        }
        log(`Fallo con ${model}: ${err}`);
        break;
      }
    }
    if (idx + 1 < models.length) {
      log(`Cambiando al modelo ${models[idx + 1]}...`);
    }
  }
  throw lastError ?? new Error("Todos los modelos fallaron");
}

// Regex que detecta el inicio de un turno de diálogo. Si se conocen los
// nombres reales de los presentadores, se restringe a ellos para no tratar
// como speaker líneas tipo "Nota:" o "12:30".
export function speakerLineRegex(speakerNames?: string[]): RegExp {
  if (speakerNames && speakerNames.length > 0) {
    const escaped = speakerNames.map((name) =>
      name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    );
    return new RegExp(`^\\s*(${escaped.join("|")})\\s*:`);
  }
  return /^\s*[\wÁÉÍÓÚÑáéíóúñ]+\s*:/;
}

// Divide el libreto en segmentos de turnos de diálogo completos (~maxChars)
// para generar el audio por partes: progreso real y reproducción anticipada.
export function splitScriptIntoSegments(
  script: string,
  maxChars = TTS_SEGMENT_MAX_CHARS,
  speakerNames?: string[]
): string[] {
  const speakerRegex = speakerLineRegex(speakerNames);
  const lines = script.split("\n");
  const turns: string[] = [];
  let current = "";
  for (const line of lines) {
    if (speakerRegex.test(line) && current.trim()) {
      turns.push(current.trim());
      current = line;
    } else {
      current += (current ? "\n" : "") + line;
    }
  }
  if (current.trim()) turns.push(current.trim());

  const segments: string[] = [];
  let segment = "";
  for (const turn of turns) {
    if (segment && segment.length + turn.length + 1 > maxChars) {
      segments.push(segment);
      segment = turn;
    } else {
      segment += (segment ? "\n" : "") + turn;
    }
  }
  if (segment.trim()) segments.push(segment);
  return segments.length > 0 ? segments : [script];
}
