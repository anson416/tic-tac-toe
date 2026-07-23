// model-config-controller.ts — reads the model-config inputs, validates them,
// and sends a ReconfigureReq on Apply (req 4). Also surfaces the live config
// to other components (e.g. initial net shape). Rebuilds only on explicit
// Apply, never on every keystroke.
import { byId } from "../by-id";
import type { WorkerBackendController } from "../worker-client";
import type { ChartsController } from "./charts-controller";
import type { NetConfig } from "../../train-protocol";

const NEURON_TITLE =
  "How many neurons in this layer. More neurons = more capacity to learn, but slower training.";

export class ModelConfigController {
  constructor(
    private worker: WorkerBackendController,
    private charts: ChartsController,
  ) {}

  init(): void {
    // Live output labels for the range inputs.
    byId<HTMLInputElement>("cfg-learning-rate").addEventListener("input", (e) => {
      byId("lr-out").textContent = (e.target as HTMLInputElement).valueAsNumber.toFixed(4);
    });
    byId<HTMLInputElement>("cfg-discount").addEventListener("input", (e) => {
      byId("disc-out").textContent = (e.target as HTMLInputElement).valueAsNumber.toFixed(3);
    });
    byId<HTMLInputElement>("cfg-epsilon-min").addEventListener("input", (e) => {
      byId("emin-out").textContent = (e.target as HTMLInputElement).valueAsNumber.toFixed(2);
    });
    byId<HTMLInputElement>("cfg-epsilon-decay").addEventListener("input", (e) => {
      byId("edec-out").textContent = (e.target as HTMLInputElement).valueAsNumber.toFixed(5);
    });
    // Rebuild the per-layer neuron inputs when the layer count changes (req 9).
    const hiddenLayers = byId<HTMLInputElement>("cfg-hidden-layers");
    hiddenLayers.addEventListener("input", () => this.renderNeuronInputs());
    hiddenLayers.addEventListener("change", () => this.renderNeuronInputs());
    this.renderNeuronInputs();
    byId("btn-apply-config").addEventListener("click", () => this.apply());
  }

  /**
   * Populate #cfg-neurons-list with one neuron-count input per hidden layer
   * (req 9). Existing values are preserved by index; new inputs default to 64.
   */
  renderNeuronInputs(): void {
    const list = byId<HTMLDivElement>("cfg-neurons-list");
    let count = byId<HTMLInputElement>("cfg-hidden-layers").valueAsNumber;
    count = Number.isFinite(count) ? Math.floor(count) : 0;
    count = Math.max(0, Math.min(6, count));

    // Preserve current values before rebuilding.
    const prev = Array.from(
      list.querySelectorAll<HTMLInputElement>(".cfg-neuron"),
    ).map((el) => el.value);

    list.replaceChildren();
    for (let i = 0; i < count; i++) {
      const input = document.createElement("input");
      input.className = "num cfg-neuron";
      input.type = "number";
      input.min = "1";
      input.max = "512";
      input.step = "1";
      input.style.width = "60px";
      input.title = NEURON_TITLE;
      input.dataset.layer = String(i);
      input.value = prev[i] ?? "64";
      list.appendChild(input);
    }
  }

  /** Read + validate the current inputs into a NetConfig. Throws on bad input. */
  read(): NetConfig {
    const lr = byId<HTMLInputElement>("cfg-learning-rate").valueAsNumber;
    const discount = byId<HTMLInputElement>("cfg-discount").valueAsNumber;
    const batchSize = byId<HTMLInputElement>("cfg-batch-size").valueAsNumber;
    const bufferSize = byId<HTMLInputElement>("cfg-buffer-size").valueAsNumber;
    const targetSync = byId<HTMLInputElement>("cfg-target-sync").valueAsNumber;
    const minEpsilon = byId<HTMLInputElement>("cfg-epsilon-min").valueAsNumber;
    const epsilonDecay = byId<HTMLInputElement>("cfg-epsilon-decay").valueAsNumber;

    // One neuron-count input per hidden layer (req 9). The layer count is just
    // the number of inputs present in the list.
    const neuronInputs = byId<HTMLDivElement>("cfg-neurons-list").querySelectorAll<HTMLInputElement>(
      ".cfg-neuron",
    );
    const hiddenLayers = Array.from(neuronInputs, (el) => {
      const neurons = el.valueAsNumber;
      if (!Number.isFinite(neurons) || neurons < 1 || neurons > 512) {
        throw new Error("Neurons per layer must be 1–512.");
      }
      return { neurons: Math.floor(neurons) };
    });

    if (!Number.isFinite(lr) || lr <= 0 || lr > 1) {
      throw new Error("Learning rate must be in (0, 1].");
    }
    if (!Number.isFinite(discount) || discount < 0 || discount > 1) {
      throw new Error("Discount γ must be in [0, 1].");
    }
    if (!Number.isFinite(batchSize) || batchSize < 1) {
      throw new Error("Batch size must be ≥ 1.");
    }

    return {
      hiddenLayers,
      learningRate: lr,
      discount,
      batchSize: Math.floor(batchSize),
      bufferSize: Math.floor(bufferSize),
      targetSyncEvery: Math.floor(targetSync),
      minEpsilon,
      epsilonDecay,
    };
  }

  apply(): void {
    let cfg: NetConfig;
    try {
      cfg = this.read();
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
      return;
    }
    // Fire-and-forget; the worker resets its metrics, which the UI picks up
    // via the metrics listener. `control` assigns the request id.
    this.worker.control({ config: cfg });
    // Rebuilding the model invalidates the old training curves (req 7).
    this.charts.reset();
  }
}
