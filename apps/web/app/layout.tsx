import "@fontsource-variable/plus-jakarta-sans";

import { GeistMono } from "geist/font/mono";
import type { Metadata } from "next";

import { SiteFooter } from "@/components/common/SiteFooter";
import { SiteHeader } from "@/components/common/SiteHeader";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "HerbaScope X - Microscopic Screening", template: "%s · HerbaScope X" },
  description:
    "Evidence-driven preliminary visual screening of microscopic medicinal-plant material: classification, reference retrieval, unknown detection and a deterministic decision.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${GeistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:shadow"
        >
          Skip to content
        </a>
        <SiteHeader />
        <main id="main" className="flex-1">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
