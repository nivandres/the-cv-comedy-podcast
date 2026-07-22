// Subtítulos aproximados del episodio: el TTS de Gemini no devuelve
// timestamps, así que el tiempo de cada parte se reparte proporcionalmente
// a la longitud del texto de cada turno de diálogo.
import { speakerLineRegex } from "@/lib/gemini";

export interface SubtitleTurn {
  speaker: string;
  text: string;
  weight: number;
}

// Divide el texto de un segmento en turnos de diálogo ("Alex: ...").
// El peso (longitud) permite estimar qué turno suena en cada momento.
// El texto previo al primer speaker (título, acotaciones) también cuenta:
// el TTS lo lee, así que ocupa tiempo en el audio.
export function parseTurns(
  segmentText: string,
  speakerNames: string[]
): SubtitleTurn[] {
  const speakerRegex = speakerLineRegex(speakerNames);
  const turns: SubtitleTurn[] = [];
  const clean = (text: string) =>
    text.replace(/\[(\.\.\.|énfasis)\]/gi, "").trim();
  for (const line of segmentText.split("\n")) {
    if (!line.trim()) continue;
    if (speakerRegex.test(line)) {
      const colon = line.indexOf(":");
      turns.push({
        speaker: line.slice(0, colon).trim(),
        text: clean(line.slice(colon + 1)),
        weight: Math.max(line.length, 1),
      });
    } else if (turns.length > 0) {
      turns[turns.length - 1].text += " " + clean(line);
      turns[turns.length - 1].weight += line.length;
    } else {
      turns.push({
        speaker: "",
        text: clean(line),
        weight: Math.max(line.length, 1),
      });
    }
  }
  return turns.length > 0
    ? turns
    : [{ speaker: "", text: segmentText.trim(), weight: 1 }];
}

// Elige el turno que suena en `fraction` (0-1) del segmento, repartiendo el
// tiempo proporcionalmente al peso de cada turno.
export function pickTurn(
  turns: SubtitleTurn[],
  fraction: number
): SubtitleTurn | null {
  if (turns.length === 0) return null;
  const totalWeight = turns.reduce((sum, turn) => sum + turn.weight, 0);
  let target = Math.max(0, Math.min(1, fraction)) * totalWeight;
  for (const turn of turns) {
    if (target <= turn.weight) return turn;
    target -= turn.weight;
  }
  return turns[turns.length - 1];
}
