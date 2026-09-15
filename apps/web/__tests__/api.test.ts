import { afterEach, describe, expect, it, vi } from "vitest";

import { analyzeImage, API_BASE_URL, ApiError, apiUrl, getAnalysis } from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("api client", () => {
  it("builds absolute URLs from API-relative paths", () => {
    expect(apiUrl("/reference/REF0001/image")).toBe(`${API_BASE_URL}/reference/REF0001/image`);
  });

  it("posts the image as multipart form data under the `image` field", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { id: "abc" }));
    vi.stubGlobal("fetch", fetchMock);
    const sample = new File(["x"], "s.png", { type: "image/png" });
    await analyzeImage(sample);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE_URL}/analyze`);
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("image")).toBe(sample);
  });

  it("surfaces the API detail message and status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(415, { detail: "Unsupported media type" })));
    await expect(getAnalysis("x")).rejects.toMatchObject({ status: 415, message: "Unsupported media type" });
  });

  it("joins FastAPI validation errors", async () => {
    const detail = [{ msg: "String should match pattern" }, { msg: "Field required" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(422, { detail })));
    await expect(getAnalysis("x")).rejects.toMatchObject({ message: "String should match pattern; Field required" });
  });

  it("reports an unreachable API with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await getAnalysis("x").catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 0 });
  });
});
