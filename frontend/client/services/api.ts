import axios from "axios";
import type { AuthResponse, PredictResult, ResultDetail, ResultSummary, User } from "@/types/api";
import { mocks } from "@/services/mocks";
import { useAuthStore } from "@/store/auth";

// Use VITE_API_BASE_URL if set, otherwise use localhost for dev or window origin for production
const baseURL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? "http://localhost:8001" : window.location.origin);
const USE_MOCKS = (import.meta.env.VITE_USE_MOCKS ?? "false") === "true";

// Debug logging
console.log('API Config:', { baseURL, USE_MOCKS, env: import.meta.env.VITE_USE_MOCKS });

const http = axios.create({ baseURL });

http.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    if (config.headers) {
      (config.headers as any).Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

export async function signup(body: { name: string; email: string; password: string }): Promise<AuthResponse> {
  if (USE_MOCKS) return mocks.signup(body);
  const { data } = await http.post<AuthResponse>("/api/auth/signup", body);
  return data;
}

export async function login(body: { email: string; password: string }): Promise<AuthResponse> {
  if (USE_MOCKS) return mocks.login(body);
  const { data } = await http.post<AuthResponse>("/api/auth/login", body);
  return data;
}

export async function me(): Promise<User> {
  if (USE_MOCKS) return mocks.me();
  const { data } = await http.get<User>("/api/me");
  return data;
}

export async function predict(files: { rgb: File; nir: File }): Promise<PredictResult> {
  if (USE_MOCKS) return mocks.predict(files);
  const form = new FormData();
  form.append("rgb", files.rgb);
  form.append("nir", files.nir);
  const { data } = await http.post<PredictResult>("/api/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listResults(): Promise<ResultSummary[]> {
  if (USE_MOCKS) return mocks.listResults();
  const { data } = await http.get<ResultSummary[]>("/api/results");
  return data;
}

export async function saveResult(resultId: string): Promise<{ ok: boolean }> {
  if (USE_MOCKS) return mocks.saveResult(resultId);
  const { data } = await http.post<{ ok: boolean }>("/api/results/save", { resultId });
  return data;
}

export async function getResult(id: string): Promise<ResultDetail> {
  if (USE_MOCKS) return mocks.getResult(id);
  const { data } = await http.get<ResultDetail>(`/api/results/${id}`);
  return data;
}
