import { describe, it, expect } from "vitest";
import ar from "@/i18n/messages/ar";
import bg from "@/i18n/messages/bg";
import cs from "@/i18n/messages/cs";
import de from "@/i18n/messages/de";
import el from "@/i18n/messages/el";
import en from "@/i18n/messages/en";
import es from "@/i18n/messages/es";
import fa from "@/i18n/messages/fa";
import fi from "@/i18n/messages/fi";
import fil from "@/i18n/messages/fil";
import fr from "@/i18n/messages/fr";
import he from "@/i18n/messages/he";
import hi from "@/i18n/messages/hi";
import hr from "@/i18n/messages/hr";
import id from "@/i18n/messages/id";
import itMessages from "@/i18n/messages/it";
import ja from "@/i18n/messages/ja";
import ko from "@/i18n/messages/ko";
import ms from "@/i18n/messages/ms";
import no from "@/i18n/messages/no";
import pl from "@/i18n/messages/pl";
import pt from "@/i18n/messages/pt";
import ro from "@/i18n/messages/ro";
import ru from "@/i18n/messages/ru";
import sq from "@/i18n/messages/sq";
import sv from "@/i18n/messages/sv";
import th from "@/i18n/messages/th";
import tr from "@/i18n/messages/tr";
import vi from "@/i18n/messages/vi";
import zhHans from "@/i18n/messages/zh-Hans";
import zhHant from "@/i18n/messages/zh-Hant";
import { LOCALES } from "@/i18n/locales";

// Claves planas de un árbol de traducciones ("a.b.c")
function flatKeys(node: unknown, prefix = ""): string[] {
  if (typeof node !== "object" || node === null) return [prefix];
  return Object.entries(node).flatMap(([key, value]) =>
    flatKeys(value, prefix ? `${prefix}.${key}` : key)
  );
}

// Todos los árboles menos es (la fuente de verdad)
const OTHER_LOCALES = {
  ar,
  bg,
  cs,
  de,
  el,
  en,
  fa,
  fi,
  fil,
  fr,
  he,
  hi,
  hr,
  id,
  it: itMessages,
  ja,
  ko,
  ms,
  no,
  pl,
  pt,
  ro,
  ru,
  sq,
  sv,
  th,
  tr,
  vi,
  "zh-Hans": zhHans,
  "zh-Hant": zhHant,
} as const;

const esKeys = flatKeys(es).sort();

it("cada locale declarado en LOCALES tiene su árbol de mensajes", () => {
  const covered = new Set(["es", ...Object.keys(OTHER_LOCALES)]);
  expect([...LOCALES].sort()).toEqual([...covered].sort());
});

describe.each(Object.entries(OTHER_LOCALES))("locale %s", (_name, tree) => {
  it("tiene exactamente las mismas claves que es (fuente de verdad)", () => {
    expect(flatKeys(tree).sort()).toEqual(esKeys);
  });

  it("conserva los placeholders {variable} de cada mensaje", () => {
    const placeholders = (value: unknown) =>
      typeof value === "string"
        ? (value.match(/\{[a-zA-Z]+\}/g) ?? []).sort()
        : [];
    const walk = (a: unknown, b: unknown, path = ""): void => {
      if (typeof a === "string") {
        expect(placeholders(b), `placeholders en ${path}`).toEqual(
          placeholders(a)
        );
        return;
      }
      if (typeof a === "object" && a !== null) {
        for (const key of Object.keys(a)) {
          walk(
            (a as Record<string, unknown>)[key],
            (b as Record<string, unknown>)?.[key],
            path ? `${path}.${key}` : key
          );
        }
      }
    };
    walk(es, tree);
  });

  it("mantiene los tokens de control del prompt (Alex:/Sam:/[énfasis]/{date})", () => {
    const script = (tree as { prompt: { script: string } }).prompt.script;
    expect(script).toContain("Alex:");
    expect(script).toContain("Sam:");
    expect(script).toContain("[énfasis]");
    expect(script).toContain("{date}");
  });
});

describe("prompt por locale (via loaders de intl-t)", () => {
  it("interpola {date} en todos los idiomas y conserva la marca", async () => {
    const { loadLocale } = await import("@/i18n/translation");
    const date = new Date("2026-07-22T12:00:00Z");
    for (const locale of LOCALES) {
      const tl = await loadLocale(locale);
      // Igual que la app: fecha gregoriana con año en dígitos latinos (para que
      // fa no salga en Jalali ni th en calendario budista). intl-t pasa los
      // strings tal cual, así que el año llega comparable con las fechas del CV.
      const dateStr = new Intl.DateTimeFormat(locale, {
        calendar: "gregory",
        numberingSystem: "latn",
        dateStyle: "long",
      }).format(date);
      const out = String(tl.prompt.script({ date: dateStr }));
      expect(out, `${locale}: quedó {date} sin interpolar`).not.toContain(
        "{date}"
      );
      expect(out, `${locale}: no aparece el año`).toContain("2026");
      expect(out, `${locale}: falta la marca`).toContain(
        "The CV Comedy Podcast"
      );
    }
  });
});
