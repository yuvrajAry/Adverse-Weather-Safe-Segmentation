import { NavLink, useNavigate } from "react-router-dom";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store/auth";

export function Navbar() {
  const isAuthed = useAuthStore((s) => s.isAuthed);
  const signOut = useAuthStore((s) => s.signOut);
  const navigate = useNavigate();
  return (
    <header className="border-b sticky top-0 z-40 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between">
        <nav className="flex items-center gap-6 text-sm">
          <NavLink to="/" className={({ isActive }) => cn(isActive && "text-primary font-semibold")}>Home</NavLink>
          <NavLink to="/about" className={({ isActive }) => cn(isActive && "text-primary font-semibold")}>About</NavLink>
          {isAuthed && (
            <NavLink to="/profile" className={({ isActive }) => cn(isActive && "text-primary font-semibold")}>Profile</NavLink>
          )}
        </nav>
        <div className="flex items-center gap-2">
          {!isAuthed ? (
            <Button variant="outline" onClick={() => navigate("/auth")}>Sign In</Button>
          ) : (
            <Button variant="outline" onClick={() => { signOut(); navigate("/auth"); }}>Sign Out</Button>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
