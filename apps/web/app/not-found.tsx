import { CompassIcon } from "@phosphor-icons/react/ssr";
import Link from "next/link";

import { EmptyState } from "@/components/common/EmptyState";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <EmptyState
        icon={CompassIcon}
        title="Page not found"
        description="This address does not match any HerbaScope X page."
        action={
          <Link
            href="/"
            className="press inline-flex rounded-lg border border-line bg-surface px-4 py-2 text-sm font-medium text-ink hover:bg-canvas focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            Back to home
          </Link>
        }
      />
    </div>
  );
}
