// Sistema de diseño de The CV Comedy Podcast: tres niveles de botón,
// tarjeta única, barra de progreso accesible, alertas con aria-live,
// spinner único y toggle de tema claro/oscuro.
import {
  useState,
  useEffect,
  type ReactNode,
  type ButtonHTMLAttributes,
} from "react";

const BUTTON_VARIANTS = {
  // Acción principal del paso: púrpura sólido (color original de la app)
  primary:
    "bg-purple-600 hover:bg-purple-700 text-white font-semibold shadow disabled:bg-gray-400 dark:disabled:bg-zinc-700",
  // Acciones secundarias (descargas, reintentos): pastel suave, como los badges
  secondary:
    "bg-purple-100 text-purple-800 hover:bg-purple-200 dark:bg-purple-950 dark:text-purple-200 dark:hover:bg-purple-900 disabled:opacity-40",
  // Acciones terciarias (enlaces de acción)
  ghost:
    "text-blue-600 hover:bg-blue-50 dark:text-blue-300 dark:hover:bg-zinc-800 disabled:opacity-40",
} as const;

const BUTTON_SIZES = {
  md: "px-4 py-2.5 text-sm",
  sm: "px-3 py-1.5 text-xs",
} as const;

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof BUTTON_VARIANTS;
  size?: keyof typeof BUTTON_SIZES;
}) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-xl transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-purple-500 ${BUTTON_SIZES[size]} ${BUTTON_VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none ${className}`}
    />
  );
}

// Tarjeta de paso: número, título, estado (bloqueado/activo/completado)
export type StepStatus = "locked" | "active" | "done";

export function StepCard({
  number,
  title,
  status,
  children,
}: {
  number: number;
  title: string;
  status: StepStatus;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={`Paso ${number}: ${title}`}
      className={`rounded-2xl border bg-white dark:bg-zinc-900 shadow-sm transition-all ${
        status === "active"
          ? "border-purple-200 dark:border-purple-900 shadow-lg shadow-purple-500/10"
          : "border-zinc-200 dark:border-zinc-800"
      } ${status === "locked" ? "opacity-50" : ""}`}
    >
      <header className="flex items-center gap-3 px-4 pt-4 sm:px-6 sm:pt-5">
        <span
          aria-hidden="true"
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
            status === "done"
              ? "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"
              : status === "active"
                ? "bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
          }`}
        >
          {status === "done" ? "✓" : number}
        </span>
        <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
          {title}
        </h2>
      </header>
      <div className="px-4 pb-4 pt-3 sm:px-6 sm:pb-5">{children}</div>
    </section>
  );
}

// Barra de progreso real (0-100): label persistente FUERA de la barra,
// modo indeterminado mientras no hay progreso medible, y semántica ARIA.
export function ProgressBar({
  value,
  label,
  indeterminate = false,
}: {
  value: number;
  label?: string;
  indeterminate?: boolean;
}) {
  const progress = Math.max(0, Math.min(100, value));
  return (
    <div className="w-full">
      {label && (
        <p
          aria-live="polite"
          className="mb-1 text-sm text-zinc-600 dark:text-zinc-300"
        >
          {label}
        </p>
      )}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(progress)}
        aria-label={label || "Progreso"}
        className="h-2.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700"
      >
        <div
          className={`h-full rounded-full bg-purple-500 transition-[width] duration-300 ${
            indeterminate
              ? "w-1/3 animate-pulse motion-reduce:animate-none"
              : ""
          }`}
          style={indeterminate ? undefined : { width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

export function Alert({
  children,
  tone = "error",
}: {
  children: ReactNode;
  tone?: "error" | "info";
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live="polite"
      className={`mt-3 rounded-xl border px-4 py-3 text-sm ${
        tone === "error"
          ? "border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
          : "border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-200"
      }`}
    >
      {children}
    </div>
  );
}

// Tema claro/oscuro: claro por defecto; oscuro solo si el usuario lo eligió
// (persistido en localStorage; el script de _document lo aplica pre-paint)
export function useTheme() {
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sincroniza con la clase puesta por el script anti-FOUC
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);
  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };
  return { isDark, toggle };
}

export function ThemeToggle() {
  const { isDark, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      aria-label={isDark ? "Cambiar a tema claro" : "Cambiar a tema oscuro"}
      title={isDark ? "Tema claro" : "Tema oscuro"}
      className="flex h-9 w-9 items-center justify-center rounded-full text-base text-zinc-500 transition-colors hover:bg-purple-100 hover:text-purple-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-purple-500 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
    >
      <span aria-hidden="true">{isDark ? "☀️" : "🌙"}</span>
    </button>
  );
}
