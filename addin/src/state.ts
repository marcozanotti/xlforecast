/**
 * Job reattachment via workbook custom properties (TS §7.3, hard rule 8).
 *
 * **Non-secret state only.** A custom property is part of the file: it travels with the
 * `.xlsx` when the workbook is emailed or synced to SharePoint. Storing a credential there
 * would be strictly worse than the `localStorage` hard rule 8 already forbids, which is how
 * the original wording — "state persists via workbook custom properties" — pointed at a leak
 * while sounding like a precaution.
 *
 * So this stores identifiers and nothing else. Authentication is the session cookie.
 */

export const JOB_ID_KEY = "XLF_JOB_ID";
export const DATA_ID_KEY = "XLF_DATA_ID";

/** Keys that must never be written here, checked rather than merely documented. */
const FORBIDDEN = ["token", "secret", "password", "cookie", "auth", "key", "credential"];

export function isStorableKey(key: string): boolean {
  const lower = key.toLowerCase();
  return !FORBIDDEN.some((word) => lower.includes(word));
}

export class ForbiddenPropertyError extends Error {
  constructor(key: string) {
    super(
      `Refusing to write '${key}' to a workbook custom property: it travels inside the ` +
        `.xlsx when the file is shared. Authentication uses the session cookie.`,
    );
    this.name = "ForbiddenPropertyError";
  }
}

/** The Office.js surface this module needs, so the logic is testable without Excel. */
export interface PropertyBag {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}

export interface AttachedJob {
  readonly jobId: string;
  readonly dataId: string | null;
}

export class WorkbookState {
  constructor(private readonly bag: PropertyBag) {}

  async attach(jobId: string, dataId: string | null): Promise<void> {
    await this.write(JOB_ID_KEY, jobId);
    if (dataId) await this.write(DATA_ID_KEY, dataId);
  }

  /** What lets a reopened pane rejoin a running job rather than losing it. */
  async attached(): Promise<AttachedJob | null> {
    const jobId = await this.bag.get(JOB_ID_KEY);
    if (!jobId) return null;
    return { jobId, dataId: await this.bag.get(DATA_ID_KEY) };
  }

  async detach(): Promise<void> {
    await this.bag.remove(JOB_ID_KEY);
    await this.bag.remove(DATA_ID_KEY);
  }

  private async write(key: string, value: string): Promise<void> {
    if (!isStorableKey(key)) throw new ForbiddenPropertyError(key);
    await this.bag.set(key, value);
  }
}
