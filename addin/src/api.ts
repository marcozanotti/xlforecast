/**
 * Client for the xlforecast API (TS §6).
 *
 * Two properties this file exists to enforce:
 *
 * - **Credentials are cookies, never headers or storage.** `credentials: "include"` sends the
 *   `HttpOnly; Secure; SameSite=Lax` session cookie. The pane never holds a token, so there is
 *   nothing to put in `localStorage` (hard rule 8) or in a workbook custom property, which
 *   would travel inside the `.xlsx` when the file is emailed.
 * - **Streaming is `fetch` + `ReadableStream`, never `EventSource`.** `EventSource` cannot
 *   send credentials on a cross-origin request in a way we can rely on, cannot set headers,
 *   and cannot POST. It is the reason the original design drifted toward a token the pane
 *   could hold.
 */

export interface ApiError {
  readonly message: string;
  readonly fix?: string;
  readonly column?: string;
  readonly unique_id?: string;
}

/** Carries the server's remedy through to the pane (FS §4 error-presentation rule). */
export class XlfError extends Error {
  constructor(
    message: string,
    readonly fix?: string,
    readonly status?: number,
    readonly column?: string,
  ) {
    super(message);
    this.name = "XlfError";
  }

  /** What S1–S5 render. Never a stack trace, always a remedy when the server gave one. */
  get display(): string {
    return this.fix ? `${this.message} ${this.fix}` : this.message;
  }
}

export interface JobProgress {
  readonly status: string;
  readonly progress: {
    readonly folds_total: number;
    readonly folds_done: number;
    readonly models_total: number;
    readonly models_done_in_fold: number;
    readonly current_model: string | null;
  } | null;
}

const TERMINAL = new Set(["completed", "failed", "cancelled", "quota_exhausted"]);

export function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}

export class ApiClient {
  constructor(private readonly baseUrl: string) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      // The whole credential story, in one option.
      credentials: "include",
      headers: { accept: "application/json", ...(init.headers ?? {}) },
    });
    if (!response.ok) throw await this.toError(response);
    return (await response.json()) as T;
  }

  private async toError(response: Response): Promise<XlfError> {
    let detail: ApiError | undefined;
    try {
      const body = (await response.json()) as { detail?: ApiError | string };
      detail = typeof body.detail === "string" ? { message: body.detail } : body.detail;
    } catch {
      // A non-JSON error body is still an error; the pane must not show the raw HTML of a
      // proxy's 502 page.
      detail = undefined;
    }
    return new XlfError(
      detail?.message ?? `The server returned ${response.status}.`,
      detail?.fix,
      response.status,
      detail?.column,
    );
  }

  async uploadPanel(
    parquet: ArrayBuffer,
    mapping: { uniqueIdCol: string; dsCol: string; yCol: string; freq: string; horizon: number },
  ): Promise<{ data_id: string; profile: Record<string, unknown> }> {
    const query = new URLSearchParams({
      unique_id_col: mapping.uniqueIdCol,
      ds_col: mapping.dsCol,
      y_col: mapping.yCol,
      freq: mapping.freq,
      h: String(mapping.horizon),
    });
    return this.request(`/v1/data?${query}`, {
      method: "POST",
      body: parquet,
      headers: { "content-type": "application/octet-stream" },
    });
  }

  /** S3's Run button. Mints the token `submitJob` requires (AC-503). */
  async confirm(
    dataId: string,
    request: Record<string, unknown>,
  ): Promise<{ confirmation_token: string }> {
    return this.request("/v1/confirm", {
      method: "POST",
      body: JSON.stringify({ data_id: dataId, request }),
      headers: { "content-type": "application/json" },
    });
  }

  async submitJob(body: Record<string, unknown>): Promise<{ job_id: string }> {
    return this.request("/v1/jobs", {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "content-type": "application/json" },
    });
  }

  async jobStatus(jobId: string): Promise<JobProgress> {
    return this.request(`/v1/jobs/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.request(`/v1/jobs/${jobId}`, { method: "DELETE" });
  }

  async results(jobId: string): Promise<Record<string, unknown>> {
    return this.request(`/v1/jobs/${jobId}/results`);
  }

  /**
   * Progress as server-sent events over `fetch` (TS §7.3).
   *
   * Falls back to polling where streaming is unavailable — Excel's webview differs across
   * Windows (WebView2), Mac (WKWebView) and the browser, and G5 exists because that is not
   * something to take on faith.
   */
  async *streamProgress(jobId: string, signal?: AbortSignal): AsyncGenerator<JobProgress> {
    const response = await fetch(`${this.baseUrl}/v1/jobs/${jobId}/stream`, {
      credentials: "include",
      headers: { accept: "text/event-stream" },
      ...(signal ? { signal } : {}),
    });
    if (!response.ok || !response.body) throw await this.toError(response);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line; a frame can arrive split across reads.
      let separator = buffer.indexOf("\n\n");
      while (separator !== -1) {
        const frame = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);
        const payload = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("");
        if (payload) yield JSON.parse(payload) as JobProgress;
        separator = buffer.indexOf("\n\n");
      }
    }
  }
}
