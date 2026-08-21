import { describe, expect, it } from "vitest";
import { classifyIssues, type WorkbookIssue } from "../src/sheets";
import { SHEETS } from "../src/layout";

describe("the manifest is inside the overwrite transaction (FR-703)", () => {
  it("there are five sheets, not four", () => {
    // Under the original "four sheets" wording the manifest sat outside the overwrite set,
    // so a re-run left a manifest describing the PREVIOUS run beside new results.
    expect(SHEETS).toHaveLength(5);
    expect(SHEETS).toContain("XLF_Manifest");
  });
});

describe("workbook preflight (FR-703a)", () => {
  it("a protected sheet blocks the whole write", () => {
    // Four correct sheets and one stale sheet is worse than a clean refusal: nothing on the
    // surface tells the user which is which.
    const issues: WorkbookIssue[] = [{ kind: "protected", sheet: "XLF_Leaderboard" }];
    expect(classifyIssues(issues).blocking).toHaveLength(1);
    expect(classifyIssues(issues).warnings).toHaveLength(0);
  });

  it("co-authoring blocks", () => {
    expect(classifyIssues([{ kind: "co-authoring" }]).blocking).toHaveLength(1);
  });

  it("formulas referencing an output sheet warn rather than block", () => {
    // The user may well want to overwrite anyway; they just need to know first.
    const issues: WorkbookIssue[] = [
      { kind: "referenced", sheet: "XLF_Forecast", by: ["Sheet1!B2"] },
    ];
    expect(classifyIssues(issues).blocking).toHaveLength(0);
    expect(classifyIssues(issues).warnings).toHaveLength(1);
  });

  it("a clean workbook produces neither", () => {
    expect(classifyIssues([])).toEqual({ blocking: [], warnings: [] });
  });

  it("separates blocking from warning when both are present", () => {
    const result = classifyIssues([
      { kind: "protected", sheet: "XLF_Forecast" },
      { kind: "referenced", sheet: "XLF_Leaderboard", by: ["Sheet1!A1"] },
    ]);
    expect(result.blocking).toHaveLength(1);
    expect(result.warnings).toHaveLength(1);
  });
});
