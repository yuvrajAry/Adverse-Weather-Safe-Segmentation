import { create } from "zustand";
import type { PredictResult } from "@/types/api";

export type ResultsState = {
  lastResult: PredictResult | null;
  history: PredictResult[];
  setLastResult: (r: PredictResult | null) => void;
  addToHistory: (r: PredictResult) => void;
  setHistory: (list: PredictResult[]) => void;
};

export const useResultsStore = create<ResultsState>()((set) => ({
  lastResult: null,
  history: [],
  setLastResult: (r) => set({ lastResult: r }),
  addToHistory: (r) => set((s) => ({ history: [r, ...s.history] })),
  setHistory: (list) => set({ history: list }),
}));
