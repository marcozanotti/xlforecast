import { beforeEach, describe, expect, it } from "vitest";
import {
  DATA_ID_KEY,
  ForbiddenPropertyError,
  JOB_ID_KEY,
  WorkbookState,
  isStorableKey,
  type PropertyBag,
} from "../src/state";

class FakeBag implements PropertyBag {
  readonly values = new Map<string, string>();
  async get(key: string) { return this.values.get(key) ?? null; }
  async set(key: string, value: string) { this.values.set(key, value); }
  async remove(key: string) { this.values.delete(key); }
}

let bag: FakeBag;
let state: WorkbookState;

beforeEach(() => {
  bag = new FakeBag();
  state = new WorkbookState(bag);
});

describe("reattachment (TS §7.3)", () => {
  it("lets a reopened pane rejoin a running job", async () => {
    await state.attach("job-1", "data-1");
    const reopened = await new WorkbookState(bag).attached();
    expect(reopened).toEqual({ jobId: "job-1", dataId: "data-1" });
  });

  it("reports nothing when no job is attached", async () => {
    expect(await state.attached()).toBeNull();
  });

  it("tolerates a job id without a data id", async () => {
    await state.attach("job-1", null);
    expect(await state.attached()).toEqual({ jobId: "job-1", dataId: null });
  });

  it("detaches both keys", async () => {
    await state.attach("job-1", "data-1");
    await state.detach();
    expect(bag.values.size).toBe(0);
  });
});

describe("credentials never reach the workbook (hard rule 8)", () => {
  it("stores identifiers only", async () => {
    await state.attach("job-1", "data-1");
    expect([...bag.values.keys()].sort()).toEqual([DATA_ID_KEY, JOB_ID_KEY].sort());
  });

  it.each(["XLF_TOKEN", "session_cookie", "api_key", "user_password", "auth_header", "my_secret", "credential_x"])(
    "refuses to store %s",
    (key) => {
      // A custom property is part of the file: it travels inside the .xlsx when the workbook
      // is emailed. Checked rather than merely documented, because the original spec wording
      // pointed at exactly this leak while sounding like a precaution.
      expect(isStorableKey(key)).toBe(false);
    },
  );

  it.each([JOB_ID_KEY, DATA_ID_KEY])("allows %s", (key) => {
    expect(isStorableKey(key)).toBe(true);
  });

  it("throws rather than silently skipping a forbidden write", async () => {
    const leaky = new WorkbookState(bag) as unknown as {
      write(key: string, value: string): Promise<void>;
    };
    await expect(leaky.write("XLF_TOKEN", "secret")).rejects.toThrowError(ForbiddenPropertyError);
    expect(bag.values.size).toBe(0);
  });
});
