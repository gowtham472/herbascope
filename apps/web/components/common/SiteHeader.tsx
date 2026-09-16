"use client";

import { ClockCounterClockwiseIcon, MicroscopeIcon } from "@phosphor-icons/react/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Wordmark } from "./Wordmark";

const LINKS = [
  { href: "/analyze", label: "Analyze", icon: MicroscopeIcon },
  { href: "/history", label: "History", icon: ClockCounterClockwiseIcon },
] as const;

export function SiteHeader() {
  const pathname = usePathname();
  const onMarketing = pathname === "/";

  return (
    <header className="sticky top-0 z-30 h-14 border-b border-line bg-surface/85 backdrop-blur print:hidden">
      <nav aria-label="Main" className="mx-auto flex h-full max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link
          href="/"
          aria-label="HerbaScope X home"
          className="press flex items-center gap-2 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-600"
        >
          <Wordmark className="h-6 w-auto" />
          <span className="rounded bg-ink-900 px-1.5 py-0.5 text-[10px] font-bold tracking-widest text-brand-500">X</span>
        </Link>

        <div className="flex items-center gap-1">
          <ul className="flex items-center gap-1">
            {LINKS.map(({ href, label, icon: Icon }) => {
              const active = pathname === href || pathname.startsWith(`${href}/`);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    aria-current={active ? "page" : undefined}
                    className={`press inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                      active ? "bg-ink-900 text-white" : "text-muted hover:bg-canvas hover:text-ink"
                    }`}
                  >
                    <Icon aria-hidden="true" weight={active ? "fill" : "regular"} className="size-4" />
                    {label}
                  </Link>
                </li>
              );
            })}
          </ul>
          {onMarketing ? (
            <Link
              href="/analyze"
              className="press ml-2 hidden rounded-full bg-brand-500 px-4 py-1.5 text-sm font-semibold text-ink-900 hover:bg-brand-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 sm:inline-flex"
            >
              Screen a sample
            </Link>
          ) : null}
        </div>
      </nav>
    </header>
  );
}
