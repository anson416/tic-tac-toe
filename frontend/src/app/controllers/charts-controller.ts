// charts-controller.ts — two hand-rolled SVG charts: smoothed TD loss and
// rolling win-rate vs random (req 5). Redrawn on each metrics push (≤5/sec).
import { byId } from "../by-id";
import { RingBuffer, drawChart } from "../ui/chart-svg";
import type { Metrics } from "../../nn/types";

export class ChartsController {
  private lossBuf = new RingBuffer<number>(200);
  private winBuf = new RingBuffer<number>(200);
  private svgLoss: SVGSVGElement;
  private svgWin: SVGSVGElement;

  constructor() {
    this.svgLoss = byId<SVGSVGElement>("chart-loss");
    this.svgWin = byId<SVGSVGElement>("chart-winrate");
  }

  init(): void {
    this.redraw();
  }

  onMetrics(m: Metrics): void {
    this.lossBuf.push(m.smoothedLoss);
    this.winBuf.push(m.winRateVsRandom);
    this.redraw();
  }

  /** Clear both charts — used when the model is rebuilt (req 7). */
  reset(): void {
    this.lossBuf.clear();
    this.winBuf.clear();
    this.redraw();
  }

  private redraw(): void {
    const maxLoss = Math.max(0.001, ...this.lossBuf.values);
    drawChart(this.svgLoss, this.lossBuf, {
      minY: 0,
      maxY: maxLoss * 1.1,
      formatValue: (v) => `loss ${v.toFixed(3)}`,
    });
    drawChart(this.svgWin, this.winBuf, {
      minY: 0,
      maxY: 1,
      formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    });
  }
}
