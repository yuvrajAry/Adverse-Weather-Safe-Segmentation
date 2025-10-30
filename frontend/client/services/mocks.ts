import type { AuthResponse, PredictResult, ResultDetail, ResultSummary, User } from "@/types/api";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const demo = {
  original: "/placeholder.svg",
  mask: "/placeholder.svg",
  heatmap: "/placeholder.svg",
  overlay: "/placeholder.svg",
};

let mockUser: User = { id: "u_1", name: "Demo User", email: "demo@example.com" };
let token = "mock-token";
const db: Record<string, PredictResult> = {};

function id() {
  return Math.random().toString(36).slice(2, 10);
}

export const mocks = {
  async signup({ name, email }: { name: string; email: string; password: string }): Promise<AuthResponse> {
    await delay(700);
    mockUser = { id: "u_" + id(), name, email };
    token = "t_" + id();
    return { token, user: mockUser };
  },
  async login({ email }: { email: string; password: string }): Promise<AuthResponse> {
    await delay(600);
    mockUser = { id: mockUser.id || "u_" + id(), name: mockUser.name || "Demo User", email };
    token = "t_" + id();
    return { token, user: mockUser };
  },
  async me(): Promise<User> {
    await delay(300);
    return mockUser;
  },
  async predict(_: { rgb: File; nir: File }): Promise<PredictResult> {
    await delay(1200);
    const rid = "r_" + id();
    const createdAt = new Date().toISOString();
    const result: PredictResult = {
      id: rid,
      originalUrl: demo.original,
      maskUrl: demo.mask,
      heatmapUrl: demo.heatmap,
      overlayUrl: demo.overlay,
      createdAt,
    };
    db[rid] = result;
    return result;
  },
  async listResults(): Promise<ResultSummary[]> {
    await delay(500);
    const items = Object.values(db)
      .slice(-5)
      .reverse()
      .map((r) => ({ id: r.id, thumbnailUrl: r.overlayUrl, createdAt: r.createdAt }));
    if (!items.length) {
      const rid = "r_" + id();
      const createdAt = new Date().toISOString();
      db[rid] = {
        id: rid,
        originalUrl: demo.original,
        maskUrl: demo.mask,
        heatmapUrl: demo.heatmap,
        overlayUrl: demo.overlay,
        createdAt,
      };
      return [{ id: rid, thumbnailUrl: demo.overlay, createdAt }];
    }
    return items;
  },
  async saveResult(_: string): Promise<{ ok: boolean }> {
    await delay(400);
    return { ok: true };
  },
  async getResult(id: string): Promise<ResultDetail> {
    await delay(400);
    return db[id] || {
      id,
      originalUrl: demo.original,
      maskUrl: demo.mask,
      heatmapUrl: demo.heatmap,
      overlayUrl: demo.overlay,
      createdAt: new Date().toISOString(),
    };
  },
};
