// activations.ts — ReLU + its derivative. The output layer is linear
// (passthrough) so Q-values are raw — proper DQN.

export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/** ReLU derivative: 1 where the pre-activation was > 0, else 0. */
export function reluDeriv(z: number): number {
  return z > 0 ? 1 : 0;
}
