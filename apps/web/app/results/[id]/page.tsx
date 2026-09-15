import type { Metadata } from "next";

import { ResultView } from "@/components/report/ResultView";

export const metadata: Metadata = { title: "Screening result" };

export default async function ResultPage(props: PageProps<"/results/[id]">) {
  const { id } = await props.params;
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <ResultView id={id} />
    </div>
  );
}
