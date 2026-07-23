// players-controller.ts — per-seat X/O Human/AI toggles (req 3). Each seat is a
// single button in the merged scoreboard above the board: clicking it flips
// between Human and AI. X always moves first.
export type Seat = "x" | "o";
export type SeatChoice = "human" | "ai";
export type Seats = { x: SeatChoice; o: SeatChoice };

export class PlayersController {
  private seats: Seats = { x: "human", o: "human" };
  private onChangeCb: ((s: Seats) => void) | null = null;

  init(): void {
    for (const seat of ["x", "o"] as const) {
      const btn = document.querySelector<HTMLButtonElement>(`#seat-${seat}`);
      btn?.addEventListener("click", () =>
        this.set(seat, this.seats[seat] === "human" ? "ai" : "human"),
      );
    }
    this.render();
  }

  onChange(cb: (s: Seats) => void): void {
    this.onChangeCb = cb;
  }

  get(): Seats {
    return this.seats;
  }

  set(seat: Seat, choice: SeatChoice): void {
    this.seats[seat] = choice;
    this.render();
    this.onChangeCb?.(this.seats);
  }

  /** Which seat controls the given side (X=first, O=second). */
  seatForSide(side: 1 | -1): SeatChoice {
    return side === 1 ? this.seats.x : this.seats.o;
  }

  private render(): void {
    for (const seat of ["x", "o"] as const) {
      const choice = this.seats[seat];
      const btn = document.querySelector<HTMLButtonElement>(`#seat-${seat}`);
      btn?.setAttribute("data-choice", choice);
      const label = document.querySelector<HTMLElement>(
        `.seat-mode[data-seat="${seat}"]`,
      );
      if (label) label.textContent = choice === "human" ? "Human" : "AI";
    }
  }
}
