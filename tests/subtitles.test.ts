import { describe, it, expect } from "vitest";
import { parseTurns, pickTurn } from "@/lib/subtitles";

const SPEAKERS = ["Alex", "Sam"];

describe("parseTurns", () => {
  it("separa turnos por speaker y limpia marcadores", () => {
    const turns = parseTurns(
      "Alex: Bienvenidos [...] al show.\nSam: Un CV [énfasis] especial.",
      SPEAKERS
    );
    expect(turns).toHaveLength(2);
    expect(turns[0].speaker).toBe("Alex");
    expect(turns[0].text).not.toContain("[...]");
    expect(turns[1].text).not.toContain("[énfasis]");
  });

  it("el preámbulo sin speaker cuenta como turno propio", () => {
    const turns = parseTurns(
      "[Episodio de prueba]\nAlex: hola\nSam: adiós",
      SPEAKERS
    );
    expect(turns).toHaveLength(3);
    expect(turns[0].speaker).toBe("");
    expect(turns[0].text).toContain("Episodio de prueba");
  });

  it("las líneas de continuación se acumulan en el turno anterior", () => {
    const turns = parseTurns("Alex: hola\n(se ríe)\nSam: adiós", SPEAKERS);
    expect(turns).toHaveLength(2);
    expect(turns[0].text).toContain("(se ríe)");
  });

  it("líneas tipo «Nota:» no crean speakers falsos", () => {
    const turns = parseTurns("Alex: hola\nNota: aplausos", SPEAKERS);
    expect(turns).toHaveLength(1);
    expect(turns[0].speaker).toBe("Alex");
  });

  it("texto vacío devuelve un turno neutro", () => {
    expect(parseTurns("", SPEAKERS)).toHaveLength(1);
  });
});

describe("pickTurn", () => {
  const turns = [
    { speaker: "Alex", text: "uno", weight: 10 },
    { speaker: "Sam", text: "dos", weight: 10 },
    { speaker: "Alex", text: "tres", weight: 10 },
  ];

  it("elige por fracción de tiempo ponderada", () => {
    expect(pickTurn(turns, 0)?.text).toBe("uno");
    expect(pickTurn(turns, 0.5)?.text).toBe("dos");
    expect(pickTurn(turns, 1)?.text).toBe("tres");
  });

  it("fracciones fuera de rango se acotan", () => {
    expect(pickTurn(turns, -1)?.text).toBe("uno");
    expect(pickTurn(turns, 2)?.text).toBe("tres");
  });

  it("lista vacía devuelve null", () => {
    expect(pickTurn([], 0.5)).toBeNull();
  });
});
