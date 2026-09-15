import {
  ArrowRightIcon,
  BrainIcon,
  CheckCircleIcon,
  ClockCounterClockwiseIcon,
  GavelIcon,
  ImagesIcon,
  QuestionIcon,
  ScalesIcon,
  SealCheckIcon,
  TargetIcon,
  WarningIcon,
  XCircleIcon,
} from "@phosphor-icons/react/ssr";
import Link from "next/link";

import { SystemStatus } from "@/components/common/SystemStatus";
import { HEADLINE } from "@/lib/copy";

const PIPELINE = [
  { icon: SealCheckIcon, title: "Quality & preprocessing", text: "Validates the file, measures focus, exposure and specimen visibility, then applies a fixed grayscale, no-crop transform." },
  { icon: BrainIcon, title: "DINOv2 representation", text: "A frozen DINOv2 vision transformer turns every rotation of the micrograph into an embedding, averaged into one." },
  { icon: TargetIcon, title: "Classifier", text: "Logistic regression estimates which supported class the embedding most resembles." },
  { icon: ImagesIcon, title: "Reference retrieval", text: "FAISS finds the most similar curated reference micrographs and shows them." },
  { icon: QuestionIcon, title: "Unknown assessment", text: "Distance to the reference library is compared with a threshold calibrated on real data." },
  { icon: ScalesIcon, title: "Evidence engine", text: "Compares classifier and retrieval — complementary analyses of the same representation." },
  { icon: GavelIcon, title: "Decision engine", text: "Versioned, deterministic rules. No model or LLM can override the outcome." },
];

const OUTCOMES = [
  { icon: CheckCircleIcon, tone: "border-pass-200 bg-pass-50 text-pass-700", title: "Preliminary pass", text: "Confident classifier, strong and unanimous reference support, low unknown risk and acceptable image quality." },
  { icon: WarningIcon, tone: "border-review-200 bg-review-50 text-review-700", title: "Review required", text: "Evidence is borderline or conflicting — for example the classifier and the reference library favour different classes." },
  { icon: QuestionIcon, tone: "border-unknown-200 bg-unknown-50 text-unknown-700", title: "Unknown", text: "The sample is outside the calibrated reference distribution, so no screening conclusion is drawn." },
];

const SUPPORTED = [
  "Microscopic visual screening",
  "Reference-consistency checks with visible references",
  "Unknown / out-of-distribution rejection",
  "Detection of classifier–reference disagreement",
  "Preliminary decision support",
];

const NOT_CLAIMED = [
  "Definitive authentication or laboratory replacement",
  "Chemical authentication",
  "Adulteration detection",
  "Clinical diagnosis or regulatory certification",
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
      <section className="grid items-center gap-10 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-brand-600">Microscopic medicinal-plant screening</p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">{HEADLINE}</h1>
          <p className="mt-5 max-w-2xl text-lg text-muted">
            HerbaScope X analyses a microscopic sample, compares the prediction against known reference material,
            estimates unknown risk, and decides whether the visual evidence is strong enough for a preliminary
            screening result — or whether it is not.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/analyze"
              className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 font-medium text-white shadow-sm hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            >
              Upload microscopic sample
              <ArrowRightIcon aria-hidden="true" weight="bold" className="size-4" />
            </Link>
            <Link
              href="/history"
              className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-5 py-2.5 font-medium text-ink hover:bg-canvas focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            >
              <ClockCounterClockwiseIcon aria-hidden="true" className="size-4" />
              Analysis history
            </Link>
          </div>
        </div>
        <SystemStatus />
      </section>

      <section aria-labelledby="pipeline" className="mt-20">
        <h2 id="pipeline" className="text-2xl font-semibold text-ink">How a screening decision is made</h2>
        <p className="mt-2 max-w-3xl text-muted">
          The classifier is only one component. The product is the evidence-to-decision layer around it — and every
          signal it uses is shown on the result page.
        </p>
        <ol className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map(({ icon: Icon, title, text }, index) => (
            <li key={title} className="rounded-xl border border-line bg-surface p-5">
              <div className="flex items-center gap-3">
                <span className="grid size-9 place-items-center rounded-lg bg-brand-50 text-brand-700">
                  <Icon aria-hidden="true" weight="duotone" className="size-5" />
                </span>
                <span className="font-mono text-xs text-muted">Step {index + 1}</span>
              </div>
              <h3 className="mt-3 font-semibold text-ink">{title}</h3>
              <p className="mt-1 text-sm text-muted">{text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="outcomes" className="mt-20">
        <h2 id="outcomes" className="text-2xl font-semibold text-ink">Three possible outcomes</h2>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {OUTCOMES.map(({ icon: Icon, tone, title, text }) => (
            <div key={title} className={`rounded-xl border p-5 ${tone}`}>
              <Icon aria-hidden="true" weight="duotone" className="size-7" />
              <h3 className="mt-3 font-semibold">{title}</h3>
              <p className="mt-1 text-sm text-ink/80">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section aria-labelledby="boundary" className="mt-20 grid gap-4 md:grid-cols-2">
        <h2 id="boundary" className="sr-only">Claim boundary</h2>
        <div className="rounded-xl border border-line bg-surface p-6">
          <h3 className="font-semibold text-ink">What HerbaScope X supports</h3>
          <ul className="mt-3 space-y-2 text-sm text-muted">
            {SUPPORTED.map((item) => (
              <li key={item} className="flex gap-2">
                <CheckCircleIcon aria-hidden="true" weight="fill" className="mt-0.5 size-4 shrink-0 text-pass-600" />
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl border border-line bg-surface p-6">
          <h3 className="font-semibold text-ink">What it does not claim</h3>
          <ul className="mt-3 space-y-2 text-sm text-muted">
            {NOT_CLAIMED.map((item) => (
              <li key={item} className="flex gap-2">
                <XCircleIcon aria-hidden="true" weight="fill" className="mt-0.5 size-4 shrink-0 text-fail-600" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}
