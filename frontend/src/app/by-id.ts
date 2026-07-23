// by-id.ts — tiny typed DOM accessor used throughout the controllers.
// Constrained to Element so it works for both HTML and SVG elements.
export function byId<T extends Element = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as unknown as T;
}
