export type User = {
  id: string;
  name: string;
  email: string;
};

export type AuthResponse = {
  token: string;
  user: User;
};

export type PredictResult = {
  id: string;
  originalUrl: string;
  maskUrl: string;
  heatmapUrl: string;
  overlayUrl: string;
  createdAt: string; // ISO timestamp
};

export type ResultSummary = {
  id: string;
  thumbnailUrl: string;
  createdAt: string;
};

export type ResultDetail = PredictResult;
