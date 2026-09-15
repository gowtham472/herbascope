"use client"; // Error boundaries must be Client Components

import { useEffect } from "react";

import { ErrorState } from "@/components/common/ErrorState";

export default function Error({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <ErrorState
        title="Something went wrong in the interface"
        message="The page failed to render. Screening results already stored by the API are not affected."
        onRetry={retry}
      />
    </div>
  );
}
