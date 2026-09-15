"use client";

import { ClockCounterClockwiseIcon, MicroscopeIcon } from "@phosphor-icons/react/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { BrandMark } from "./BrandMark";

const LINKS = [
  { href: "/analyze", label: "Analyze", icon: MicroscopeIcon },
  { href: "/history", label: "History", icon: ClockCounterClockwiseIcon },
] as const;

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/90 backdrop-blur print:hidden">
      <nav aria-label="Main" className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 rounded-md font-semibold text-ink focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-600"
        >
          <BrandMark className="size-7" />
          <span>
            HerbaScope <span className="text-brand-600">X</span>
          </span>
        </Link>
        <ul className="flex items-center gap-1">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href}>
                <Link
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={`press inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-brand-600 ${
                    active ? "bg-brand-50 text-brand-700" : "text-muted hover:bg-canvas hover:text-ink"
                  }`}
                >
                  <Icon aria-hidden="true" weight={active ? "fill" : "regular"} className="size-4" />
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
