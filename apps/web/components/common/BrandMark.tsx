/** HerbaScope X mark: a leaf seen through a lens. Decorative; the wordmark carries the name. */
export function BrandMark({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={className}>
      <rect width="32" height="32" rx="8" fill="#1b5b47" />
      <circle cx="14" cy="14" r="8.5" fill="none" stroke="#d3ebdf" strokeWidth="2.2" />
      <path d="M20.5 20.5 26 26" stroke="#d3ebdf" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M10 17.5c0-5 3.2-8 8-8.2-.2 4.8-3.2 8.2-8 8.2Z" fill="#7fd1ad" />
      <path d="M10.5 17c2.2-2.2 4-3.6 6-4.8" stroke="#1b5b47" strokeWidth="1" strokeLinecap="round" fill="none" />
    </svg>
  );
}
