import { describe, expect, it } from "vitest";
import {
  encodePanel,
  excelSerialToISO,
  isExcelSerialDate,
  normaliseDate,
  normaliseNumber,
} from "../src/encode";

describe("Excel serial dates", () => {
  it("converts the epoch correctly", () => {
    // Serial 1 is 1900-01-01 under Excel's day-zero of 1899-12-30.
    expect(excelSerialToISO(1)).toBe("1899-12-31");
    expect(excelSerialToISO(45658)).toBe("2025-01-01");
  });

  it("recognises a plausible serial", () => {
    expect(isExcelSerialDate(45658)).toBe(true);
    expect(isExcelSerialDate("2025-01-01")).toBe(false);
    expect(isExcelSerialDate(null)).toBe(false);
  });

  it("converts serials rather than passing them through as numbers", () => {
    // The detail most likely to corrupt a panel silently: a serial read as a number gives a
    // valid-looking column of five-digit integers that profiles as yearly data from 1970.
    expect(normaliseDate(45658)).toBe("2025-01-01");
  });

  it("passes date strings through for the server to judge", () => {
    expect(normaliseDate("2025-01-31")).toBe("2025-01-31");
    expect(normaliseDate(" 2025-01-31 ")).toBe("2025-01-31");
  });

  it("treats blanks as missing", () => {
    expect(normaliseDate("")).toBeNull();
    expect(normaliseDate(null)).toBeNull();
    expect(normaliseDate(undefined)).toBeNull();
  });
});

describe("numbers", () => {
  it("passes finite numbers through", () => {
    expect(normaliseNumber(42.5)).toBe(42.5);
  });

  it("parses text-formatted cells", () => {
    expect(normaliseNumber("1,234.5")).toBe(1234.5);
    expect(normaliseNumber(" 17 ")).toBe(17);
  });

  it("treats blanks and non-numbers as missing rather than zero", () => {
    // Zero-filling here would manufacture intermittency and change which models the server
    // routes the series to (FR-106).
    expect(normaliseNumber("")).toBeNull();
    expect(normaliseNumber("n/a")).toBeNull();
    expect(normaliseNumber(Infinity)).toBeNull();
  });
});

describe("encodePanel", () => {
  const header = ["sku", "week", "units"];
  const mapping = { uniqueIdCol: "sku", dsCol: "week", yCol: "units" };

  it("produces an Arrow IPC stream", () => {
    const { bytes, rowCount } = encodePanel(
      header,
      [["A", 45658, 10], ["A", 45665, 12]],
      mapping,
    );
    expect(rowCount).toBe(2);
    // The stream framing begins with a continuation marker, not the file format's ARROW1.
    expect(Array.from(bytes.slice(0, 4))).toEqual([255, 255, 255, 255]);
  });

  it("skips rows with no id or no date, and counts them", () => {
    const { rowCount, skipped } = encodePanel(
      header,
      [["A", 45658, 10], ["", 45665, 12], ["B", "", 3], ["B", 45658, 5]],
      mapping,
    );
    expect(rowCount).toBe(2);
    // Counted rather than silently dropped: a discrepancy between what the user selected and
    // what was profiled must be visible (FS section 6).
    expect(skipped).toBe(2);
  });

  it("names a column that is not in the selection", () => {
    expect(() =>
      encodePanel(header, [], { ...mapping, yCol: "revenue" }),
    ).toThrowError(/revenue/);
  });

  it("handles a panel with no data rows", () => {
    expect(encodePanel(header, [], mapping).rowCount).toBe(0);
  });

  it("keeps missing values missing rather than zero-filling", () => {
    const { rowCount } = encodePanel(header, [["A", 45658, ""]], mapping);
    expect(rowCount).toBe(1);
  });
});
