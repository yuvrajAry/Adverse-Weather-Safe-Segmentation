export function LoadingSpinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2" role="status" aria-label={label}>
      <svg className="h-4 w-4 animate-spin text-primary" viewBox="0 0 50 50">
        <circle cx="25" cy="25" r="20" stroke="currentColor" strokeWidth="5" fill="none" className="opacity-30" />
        <circle cx="25" cy="25" r="20" stroke="currentColor" strokeWidth="5" fill="none" strokeDasharray="100" strokeDashoffset="75" />
      </svg>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  );
}
