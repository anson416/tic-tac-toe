// heatmap-controller.ts — per-layer activation grid with type-aware coloring
// (req task 8). Adapts to whatever layer/neuron counts the network has.
// Input cells are colored by X/O meaning, hidden cells by activation magnitude
// (yellow), and the output row is tinted by the AI's side with the softmax
// probability shown on hover (title) so the user sees which move the net favors.
// update() is called after every AI move.
import { byId } from "../by-id";
import {
  hiddenCellColor,
  inputCellColor,
  maxAbs,
  outputCellColor,
  softmax,
} from "../ui/colormap";

const LAYER_LABELS = (i: number, total: number, count: number): string => {
  if (i === 0) return "Input (10)";
  if (i === total - 1) return "Output Q (9)";
  return `Hidden ${i} (${count})`;
};

export class HeatmapController {
  private container: HTMLElement;
  private current: number[][] | null = null;
  private side: 1 | -1 = 1; // AI's side, used for output-row coloring

  constructor() {
    this.container = byId("heatmap");
  }

  init(): void {
    this.renderEmpty();
  }

  update(activations: number[][], side: 1 | -1): void {
    this.current = activations;
    this.side = side;
    this.render();
  }

  private renderEmpty(): void {
    this.container.innerHTML =
      '<span class="hint">Play a move as/against the AI to see neuron activations.</span>';
  }

  private render(): void {
    if (!this.current) {
      this.renderEmpty();
      return;
    }
    const acts = this.current;
    const total = acts.length;
    // Build as an HTML string (small node count) then set once.
    let html = "";
    for (let li = 0; li < total; li++) {
      const layer = acts[li]!;
      const isInput = li === 0;
      const isOutput = li === total - 1;
      // Output row: softmax over raw Q-values for probabilities + normalization.
      const probs = isOutput ? softmax(layer) : [];
      const maxProb = isOutput ? Math.max(...probs) : 0;
      const layerMax = maxAbs(layer); // per-layer normalization for hidden rows
      html += `<div class="hm-layer"><div class="hm-label">${LAYER_LABELS(li, total, layer.length)}</div><div class="hm-row">`;
      for (let i = 0; i < layer.length; i++) {
        const v = layer[i]!;
        let color: string;
        let title: string;
        let label = "";
        if (isInput) {
          color = inputCellColor(v);
          const meaning = v > 0 ? "X" : v < 0 ? "O" : "empty";
          title = `cell ${i}: ${meaning}`;
        } else if (isOutput) {
          const p = probs[i]!;
          color = outputCellColor(p, maxProb, this.side);
          title = `cell ${i}: Q=${v.toFixed(3)}, p=${p.toFixed(2)}`;
        } else {
          color = hiddenCellColor(v, layerMax);
          title = `n${i}: ${v.toFixed(2)}`;
        }
        html += `<div class="hm-cell${isOutput ? " hm-out" : ""}" style="background:${color}" title="${title}">${label}</div>`;
      }
      html += "</div></div>";
    }
    this.container.innerHTML = html;
  }
}
