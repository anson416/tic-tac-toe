// train.worker.ts — the training + inference backend, entirely off the main
// thread so the UI never freezes. Owns ONE weight set (online net) plus a
// periodically-synced target net and a replay buffer. Self-play generates
// transitions; trainStep samples a batch and applies one Adam update. The
// worker also answers inference requests (temperature-based move selection),
// reading the live online weights so the AI gets smarter as it trains.
//
// The worker is pure `onmessage` glue over the shared nn/game modules; its
// `self` is hand-typed because tsconfig's lib is [DOM] only (no WebWorker,
// which would clash with DOM's onmessage/postMessage).
import {
  applyMove,
  isTerminal,
  legalMoves,
  newGame,
  winner,
  type BoardState,
  type Side,
  X,
  O,
} from "../game/model";
import { encode } from "../game/encoding";
import { buildNetwork, type GradBuffers, type Network } from "../nn/network";
import { ReplayBuffer } from "../nn/replay";
import {
  defaultConfig,
  isInferenceReq,
  isReconfigureReq,
  isStartReq,
  isStopReq,
  type InferenceReq,
  type InferenceResp,
  type Metrics,
  type WorkerReq,
  type WorkerResp,
} from "../train-protocol";
import type { NetConfig, Transition } from "../nn/types";

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerReq>) => void) | null;
  postMessage: (msg: WorkerResp) => void;
};

// --- state -----------------------------------------------------------------
let config: NetConfig = defaultConfig();
let online: Network = buildNetwork(config.hiddenLayers.map((l) => l.neurons));
let target: Network = buildNetwork(config.hiddenLayers.map((l) => l.neurons));
target.copyWeightsFrom(online);
let buffer = new ReplayBuffer(config.bufferSize);
let grads: GradBuffers = online.newGradBuffers();

let training = false;
let gamesPlayed = 0;
let stepCount = 0;
let epsilon = 1;
let smoothedLoss = 0;
let winRateVsRandom = 0;
let lastReport = 0;

const TRAIN_STEPS_PER_GAME = 4;
const GAMES_PER_TICK = 8; // self-play games per scheduler tick before yielding
const EVAL_EVERY = 200; // games between win-rate evals
const EVAL_GAMES = 50; // games vs random per eval
const TICK_DELAY_MS = 0; // yield to the event loop between ticks (serves inference)

// --- helpers ---------------------------------------------------------------

function legalMaskFromCells(cells: ReadonlyArray<number>): boolean[] {
  const mask = new Array<boolean>(9).fill(false);
  for (let i = 0; i < 9; i++) if (cells[i] === 0) mask[i] = true;
  return mask;
}

/** terminalReward for `side`: +1 win, -1 loss, 0 draw/non-terminal. Zero-sum. */
function terminalReward(board: BoardState, side: Side): number {
  const w = winner(board);
  if (w === null) return 0;
  return w === side ? 1 : -1;
}

/**
 * epsilon-greedy move selection during self-play, on the ONLINE net, with
 * illegal moves masked to -Inf. Returns the chosen cell index.
 */
function selectEpsilonGreedy(board: BoardState, eps: number): number {
  const legal = legalMoves(board);
  if (legal.length === 0) throw new Error("no legal moves");
  if (Math.random() < eps) return legal[(Math.random() * legal.length) | 0];
  const q = online.qValues(encode(board.cells, board.turn));
  let best = legal[0]!;
  let bestQ = -Infinity;
  for (const a of legal) {
    if (q[a]! > bestQ) {
      bestQ = q[a]!;
      best = a;
    }
  }
  return best;
}

/** Greedy move (for eval vs random + smartness=100%). */
function greedyMove(board: BoardState): number {
  const legal = legalMoves(board);
  const q = online.qValues(encode(board.cells, board.turn));
  let best = legal[0]!;
  let bestQ = -Infinity;
  for (const a of legal) {
    if (q[a]! > bestQ) {
      bestQ = q[a]!;
      best = a;
    }
  }
  return best;
}

/** Play one self-play game, storing transitions. Returns the winner Side | null (draw). */
function playOneSelfPlayGame(): Side | null {
  let board = newGame();
  const transitions: Transition[] = [];
  let result: Side | null = null;
  while (!isTerminal(board)) {
    const side = board.turn;
    const state = encode(board.cells, side);
    const action = selectEpsilonGreedy(board, epsilon);
    const next = applyMove(board, action);
    const done = isTerminal(next);
    const reward = done ? terminalReward(next, side) : 0;
    // Negamax: after `side` moves it is the OPPONENT's turn, so the next
    // decision state must be encoded from the opponent's perspective.
    const nextState = done ? state : encode(next.cells, -side as Side);
    transitions.push({
      state,
      action,
      reward,
      nextState,
      done,
      sideFlag: side,
    });
    board = next;
    if (done) result = winner(next);
  }
  for (const t of transitions) buffer.push(t);
  return result;
}

/** One gradient step over a random mini-batch. Returns mean loss (or null if too few samples). */
function trainStep(): number | null {
  if (buffer.size < config.batchSize) return null;
  const batch = buffer.sample(config.batchSize);
  let totalLoss = 0;
  for (const t of batch) {
    const { cache } = online.forward(t.state);
    let targetVal = t.reward;
    if (!t.done) {
      const qt = target.qValues(t.nextState);
      const mask = legalMaskFromCells(t.nextState.slice(0, 9));
      let maxNext = -Infinity;
      for (let i = 0; i < 9; i++)
        if (mask[i] && qt[i]! > maxNext) maxNext = qt[i]!;
      if (maxNext === -Infinity) maxNext = 0; // no legal moves (shouldn't happen if !done)
      // Negamax bootstrap (zero-sum, single shared net): the value to the mover
      // of reaching s' is the NEGATIVE of the opponent's best value at s'. With
      // a `+` the net assumes it moves again and Q-values diverge (loss blows
      // up, win-rate stalls at ~random).
      targetVal = t.reward - config.discount * maxNext;
    }
    totalLoss += online.accumulateGrads(cache, t.action, targetVal, grads);
  }
  online.applyAdam(grads, config.learningRate, batch.length);
  stepCount += 1;
  if (stepCount % config.targetSyncEvery === 0) target.copyWeightsFrom(online);
  return totalLoss / batch.length;
}

/** Winning rate of the greedy online net vs a random opponent over EVAL_GAMES games. */
function evalVsRandom(): number {
  let wins = 0;
  for (let g = 0; g < EVAL_GAMES; g++) {
    // alternate which side the net plays
    const netSide: Side = g % 2 === 0 ? X : O;
    let board = newGame();
    let result: Side | null = null;
    while (!isTerminal(board)) {
      const move =
        board.turn === netSide ? greedyMove(board) : randomMove(board);
      board = applyMove(board, move);
      if (isTerminal(board)) result = winner(board);
    }
    if (result === netSide) wins += 1;
  }
  return wins / EVAL_GAMES;
}

function randomMove(board: BoardState): number {
  const legal = legalMoves(board);
  return legal[(Math.random() * legal.length) | 0];
}

/** Temperature-based move selection for inference (smartness slider). */
function inferenceMove(req: InferenceReq): {
  move: number;
  activations: number[][];
} {
  const cells = req.board;
  const side = req.side as Side;
  const legal: number[] = [];
  for (let i = 0; i < 9; i++) if (cells[i] === 0) legal.push(i);
  const { qValues, activations } = online.forward(encode(cells, side));
  const s = req.smartness;
  let move: number;
  if (legal.length === 0) {
    // No legal move (shouldn't happen — board not terminal); return 0 defensively.
    move = 0;
  } else if (s >= 0.99) {
    // greedy: argmax over legal
    move = legal[0]!;
    let bestQ = qValues[move]!;
    for (const a of legal)
      if (qValues[a]! > bestQ) {
        bestQ = qValues[a]!;
        move = a;
      }
  } else if (s <= 0.01) {
    // uniform random over legal
    move = legal[(Math.random() * legal.length) | 0];
  } else {
    // softmax(Q_legal / τ), τ = (1 - s) * 1.0 + 0.05
    const tau = (1 - s) * 1.0 + 0.05;
    let maxQ = -Infinity;
    for (const a of legal) if (qValues[a]! > maxQ) maxQ = qValues[a]!;
    const exps = legal.map((a) => Math.exp((qValues[a]! - maxQ) / tau));
    let sum = 0;
    for (const e of exps) sum += e;
    let r = Math.random() * sum;
    move = legal[0]!;
    for (let i = 0; i < exps.length; i++) {
      r -= exps[i]!;
      if (r <= 0) {
        move = legal[i]!;
        break;
      }
    }
  }
  return { move, activations };
}

function maybeReport(now: number, force = false): void {
  const byGames = gamesPlayed % 50 === 0;
  const bySteps = stepCount % 500 === 0;
  if (!force && !(now - lastReport > 200 && (byGames || bySteps))) return;
  lastReport = now;
  const metrics: Metrics = {
    gamesPlayed,
    stepCount,
    epsilon,
    smoothedLoss,
    winRateVsRandom,
  };
  ctx.postMessage({ id: 0, metrics });
}

/** Run a bounded slice of self-play + training, then yield to the event loop
 *  so queued inference requests are served between ticks. This is what keeps
 *  the page interactive during long training (inference is never starved). */
function trainingTick(): void {
  if (!training) return;
  for (let g = 0; g < GAMES_PER_TICK && training; g++) {
    const result = playOneSelfPlayGame();
    gamesPlayed += 1;
    if (epsilon > config.minEpsilon) {
      epsilon = Math.max(epsilon * config.epsilonDecay, config.minEpsilon);
    }
    let lastLoss: number | null = null;
    for (let i = 0; i < TRAIN_STEPS_PER_GAME; i++) {
      const loss = trainStep();
      if (loss !== null) lastLoss = loss;
    }
    if (lastLoss !== null) {
      smoothedLoss =
        smoothedLoss === 0 ? lastLoss : smoothedLoss * 0.95 + lastLoss * 0.05;
    }
    if (gamesPlayed % EVAL_EVERY === 0) {
      winRateVsRandom = evalVsRandom();
    }
    void result; // winner of a self-play game isn't tracked separately
  }
  maybeReport(performance.now());
  if (training) setTimeout(trainingTick, TICK_DELAY_MS);
}

// --- reconfigure -----------------------------------------------------------
function reconfigure(cfg: NetConfig): void {
  const wasTraining = training;
  training = false;
  config = cfg;
  online = buildNetwork(config.hiddenLayers.map((l) => l.neurons));
  target = buildNetwork(config.hiddenLayers.map((l) => l.neurons));
  target.copyWeightsFrom(online);
  buffer = new ReplayBuffer(config.bufferSize);
  grads = online.newGradBuffers();
  gamesPlayed = 0;
  stepCount = 0;
  epsilon = 1;
  smoothedLoss = 0;
  winRateVsRandom = 0;
  // Reset metrics on the UI too.
  maybeReport(performance.now(), true);
  if (wasTraining) {
    training = true;
    setTimeout(trainingTick, TICK_DELAY_MS);
  }
}

// --- message dispatch ------------------------------------------------------
ctx.onmessage = (e: MessageEvent<WorkerReq>) => {
  const req = e.data;
  try {
    if (isStartReq(req)) {
      if (!training) {
        training = true;
        setTimeout(trainingTick, TICK_DELAY_MS);
      }
    } else if (isStopReq(req)) {
      training = false; // active tick exits; no new tick scheduled
    } else if (isReconfigureReq(req)) {
      reconfigure(req.config);
    } else if (isInferenceReq(req)) {
      // Served immediately. Because trainingTick yields to the event loop
      // between slices, this handler is never starved by a long training run.
      const { move, activations } = inferenceMove(req);
      const resp: InferenceResp = { id: req.id, move, activations };
      ctx.postMessage(resp);
    }
  } catch (err) {
    const errorResp: WorkerResp = {
      id: req.id ?? 0,
      error: err instanceof Error ? err.message : String(err),
    };
    ctx.postMessage(errorResp);
  }
};
