import { describe, expect, it } from "vitest";
import { spearman } from "./stats";

describe("spearman", () => {
  it("is 1 for monotonic, -1 for reversed, handles ties", () => {
    expect(spearman([1, 2, 3, 4], [10, 20, 30, 1000])).toBeCloseTo(1, 10);
    expect(spearman([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1, 10);
    // x ranks (1,2.5,2.5,4), y ranks (1,2,3,4) -> rho = 0.9486833
    expect(spearman([1, 2, 2, 3], [1, 2, 3, 4])).toBeCloseTo(0.9486833, 6);
  });
});
