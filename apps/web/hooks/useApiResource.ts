"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/lib/api";

type ResourceState<T> =
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: ApiError };

/**
 * Fetch an API resource for a key, with loading / success / error states and a retry.
 * `load` must be a stable function (e.g. an import from lib/api); `key` drives refetching.
 */
export function useApiResource<T>(key: string, load: (key: string) => Promise<T>) {
  const [attempt, setAttempt] = useState(0);
  const requestKey = `${key}#${attempt}`;
  const [settled, setSettled] = useState<{ requestKey: string; state: ResourceState<T> } | null>(null);

  useEffect(() => {
    let active = true;
    load(key).then(
      (data) => active && setSettled({ requestKey, state: { status: "success", data } }),
      (error: unknown) =>
        active &&
        setSettled({
          requestKey,
          state: {
            status: "error",
            error: error instanceof ApiError ? error : new ApiError(0, "Unexpected error while loading data."),
          },
        }),
    );
    return () => {
      active = false;
    };
  }, [key, load, requestKey]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  const state: ResourceState<T> = settled?.requestKey === requestKey ? settled.state : { status: "loading" };
  return { state, retry };
}
