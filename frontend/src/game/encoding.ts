// encoding.ts — turn a board + side into the network's 10-input vector.
//
// 9 cells (-1 = O, 0 = empty, +1 = X) + 1 side-flag (+1 if I am X, -1 if O).
// The side-flag lets ONE network play both sides: it always sees the board
// from the current player's perspective ("am I X or O?"), so it learns a
// single Q function parameterized by side.
import type { Cell, Side } from "./model";

export const INPUT_SIZE = 10;
export const OUTPUT_SIZE = 9;

/**
 * Encode a board for the network: 9 cells (-1/0/+1) + 1 side-flag.
 * Accepts plain numbers (the worker receives a `number[]` across the message
 * boundary) as well as the typed `Cell` values used internally.
 */
export function encode(cells: ReadonlyArray<number | Cell>, side: Side): number[] {
  const out = new Array<number>(INPUT_SIZE);
  for (let i = 0; i < 9; i++) out[i] = cells[i] as number;
  out[9] = side;
  return out;
}
