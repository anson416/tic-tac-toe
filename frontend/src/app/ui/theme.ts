// theme.ts — dark-first; toggle sets data-theme on <html>.
export type Theme = "dark" | "light";

export function currentTheme(): Theme {
  return (document.documentElement.getAttribute("data-theme") as Theme) || "dark";
}

export function setTheme(t: Theme): void {
  document.documentElement.setAttribute("data-theme", t);
}

export function toggleTheme(): void {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
}
