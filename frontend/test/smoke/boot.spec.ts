// boot.spec.ts — the minimal deploy gate. Boots the BUILT dist/ via
// `vite preview` (which honors the relative base "./"), then asserts the app
// reaches a ready state with the train.worker chunk loading (HTTP 200) and no
// page errors. Mirrors the reference app's smoke test shape.
import { test, expect } from "@playwright/test";

test("app boots, worker loads, no page errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const workerStatuses: number[] = [];
  page.on("response", (r) => {
    if (r.url().includes("train.worker")) workerStatuses.push(r.status());
  });

  await page.goto("./");

  // The app sets #app-status to class "status-ready" once wiring completes.
  await expect(page.locator("#app-status")).toHaveClass(/status-ready/, { timeout: 15_000 });

  // Board cells exist.
  await expect(page.locator("#cell-4")).toBeVisible();

  expect(errors).toEqual([]);
  expect(workerStatuses.some((s) => s === 200)).toBe(true);
});
