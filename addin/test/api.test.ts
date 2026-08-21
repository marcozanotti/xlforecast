import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient, XlfError, isTerminal } from "../src/api";

const BASE = "https://api.test";

function mockFetch(impl: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const spy = vi.fn(impl as never);
  vi.stubGlobal("fetch", spy);
  return spy;
}

function sse(frames: string[]): Response {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("credentials (hard rule 8)", () => {
  it("sends the session cookie on every request", async () => {
    const spy = mockFetch(() => Response.json({ job_id: "j" }));
    await new ApiClient(BASE).jobStatus("j");
    expect(spy.mock.calls[0]![1]).toMatchObject({ credentials: "include" });
  });

  it("never sets an Authorization header", async () => {
    // The pane holds no token at all, which is what removes the pressure to store one
    // somewhere it would leak.
    const spy = mockFetch(() => Response.json({}));
    await new ApiClient(BASE).jobStatus("j");
    const headers = (spy.mock.calls[0]![1] as RequestInit).headers as Record<string, string>;
    expect(Object.keys(headers).map((k) => k.toLowerCase())).not.toContain("authorization");
  });

  it("sends credentials on the progress stream too", async () => {
    const spy = mockFetch(() => sse([]));
    const iterator = new ApiClient(BASE).streamProgress("j");
    await iterator.next();
    expect(spy.mock.calls[0]![1]).toMatchObject({ credentials: "include" });
  });
});

describe("error handling (FS §4)", () => {
  it("carries the server's remedy through to the pane", async () => {
    mockFetch(() =>
      Response.json(
        { detail: { message: "Series 'SKU-17' has 40 observations, 143 required.", fix: "Shorten the horizon." } },
        { status: 400 },
      ),
    );
    await expect(new ApiClient(BASE).jobStatus("j")).rejects.toThrowError(XlfError);
    try {
      await new ApiClient(BASE).jobStatus("j");
    } catch (error) {
      expect((error as XlfError).display).toContain("Shorten the horizon.");
      expect((error as XlfError).status).toBe(400);
    }
  });

  it("keeps the offending column so the pane can name it", async () => {
    mockFetch(() =>
      Response.json({ detail: { message: "bad column", fix: "fix it", column: "revenue" } }, { status: 400 }),
    );
    try {
      await new ApiClient(BASE).uploadPanel(new ArrayBuffer(4), {
        uniqueIdCol: "a", dsCol: "b", yCol: "revenue", freq: "W", horizon: 6,
      });
    } catch (error) {
      expect((error as XlfError).column).toBe("revenue");
    }
  });

  it("does not surface a proxy's HTML error page", async () => {
    mockFetch(() => new Response("<html>502 Bad Gateway</html>", { status: 502 }));
    try {
      await new ApiClient(BASE).jobStatus("j");
    } catch (error) {
      expect((error as XlfError).message).not.toContain("<html>");
      expect((error as XlfError).message).toContain("502");
    }
  });
});

describe("streamProgress (TS §7.3)", () => {
  it("yields one event per SSE frame", async () => {
    mockFetch(() =>
      sse([
        'data: {"status":"running","progress":null}\n\n',
        'data: {"status":"completed","progress":null}\n\n',
      ]),
    );
    const seen: string[] = [];
    for await (const event of new ApiClient(BASE).streamProgress("j")) seen.push(event.status);
    expect(seen).toEqual(["running", "completed"]);
  });

  it("reassembles a frame split across reads", async () => {
    // Chunk boundaries are not frame boundaries, and a naive reader drops the event.
    mockFetch(() => sse(['data: {"status":"run', 'ning","progress":null}\n\n']));
    const seen: string[] = [];
    for await (const event of new ApiClient(BASE).streamProgress("j")) seen.push(event.status);
    expect(seen).toEqual(["running"]);
  });

  it("ignores non-data lines such as heartbeat comments", async () => {
    mockFetch(() => sse([': keep-alive\n\n', 'data: {"status":"queued","progress":null}\n\n']));
    const seen: string[] = [];
    for await (const event of new ApiClient(BASE).streamProgress("j")) seen.push(event.status);
    expect(seen).toEqual(["queued"]);
  });

  it("surfaces a failed stream as a named error", async () => {
    mockFetch(() => Response.json({ detail: { message: "no job", fix: "check the id" } }, { status: 404 }));
    const iterator = new ApiClient(BASE).streamProgress("ghost");
    await expect(iterator.next()).rejects.toThrowError(XlfError);
  });
});

describe("isTerminal", () => {
  it.each(["completed", "failed", "cancelled", "quota_exhausted"])("%s ends polling", (status) => {
    expect(isTerminal(status)).toBe(true);
  });

  it.each(["queued", "running"])("%s keeps polling", (status) => {
    expect(isTerminal(status)).toBe(false);
  });
});

describe("submission (AC-503)", () => {
  it("confirms before submitting", async () => {
    const calls: string[] = [];
    mockFetch((url) => {
      calls.push(String(url));
      return Response.json({ confirmation_token: "t", job_id: "j" });
    });
    const client = new ApiClient(BASE);
    const { confirmation_token } = await client.confirm("d1", { h: 13 });
    await client.submitJob({ data_id: "d1", confirmation_token });
    expect(calls[0]).toContain("/v1/confirm");
    expect(calls[1]).toContain("/v1/jobs");
  });
});
