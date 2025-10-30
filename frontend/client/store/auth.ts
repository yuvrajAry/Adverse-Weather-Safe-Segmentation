import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types/api";
import * as api from "@/services/api";

export type AuthState = {
  user: User | null;
  token: string | null;
  isAuthed: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => void;
  loadFromStorage: () => Promise<void>;
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthed: false,
      async signIn(email, password) {
        const res = await api.login({ email, password });
        set({ user: res.user, token: res.token, isAuthed: true });
      },
      async signUp(name, email, password) {
        const res = await api.signup({ name, email, password });
        set({ user: res.user, token: res.token, isAuthed: true });
      },
      signOut() {
        set({ user: null, token: null, isAuthed: false });
      },
      async loadFromStorage() {
        const token = get().token;
        if (!token) return;
        try {
          const me = await api.me();
          set({ user: me, isAuthed: true });
        } catch {
          set({ user: null, token: null, isAuthed: false });
        }
      },
    }),
    {
      name: "aw-safeseg-auth",
      partialize: (s) => ({ user: s.user, token: s.token, isAuthed: s.isAuthed }),
    },
  ),
);
