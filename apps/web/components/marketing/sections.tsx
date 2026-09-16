import {
  ArrowDownIcon,
  ArrowRightIcon,
  BrainIcon,
  CheckCircleIcon,
  ClockCounterClockwiseIcon,
  GavelIcon,
  GaugeIcon,
  ImagesIcon,
  QuestionIcon,
  ScalesIcon,
  SealCheckIcon,
  TargetIcon,
  WarningIcon,
  XCircleIcon,
} from "@phosphor-icons/react/ssr";
import Link from "next/link";
import type { ReactNode } from "react";

import { Mascot } from "@/components/common/Mascot";
import { SystemStatus } from "@/components/common/SystemStatus";
import { Reveal } from "@/components/motion/Reveal";
import { HEADLINE } from "@/lib/copy";

import { PinnedRail } from "./PinnedRail";
import { SnapRail } from "./SnapRail";

function Eyebrow({ children, tone = "dark" }: { children: ReactNode; tone?: "dark" | "light" }) {
  return (
    <p className={`text-xs font-semibold uppercase tracking-[0.18em] ${tone === "dark" ? "text-brand-700" : "text-brand-500"}`}>
      {children}
    </p>
  );
}

function Section({ children, tone = "light", className = "" }: { children: ReactNode; tone?: "light" | "paper" | "dark"; className?: string }) {
  const surface = { light: "bg-surface text-ink", paper: "bg-canvas text-ink", dark: "bg-ink-900 text-white" }[tone];
  return (
    <section className={`${surface} ${className}`}>
      <div className="mx-auto w-full max-w-6xl px-5 py-20 sm:px-8 lg:py-28">{children}</div>
    </section>
  );
}

export function Hero() {
  return (
    <section className="grid-field relative overflow-hidden bg-ink-900 text-white">
      <div className="mx-auto grid w-full max-w-6xl items-center gap-12 px-5 pb-20 pt-16 sm:px-8 lg:grid-cols-[1.1fr_0.9fr] lg:pb-28 lg:pt-24">
        <div>
          <p className="fade-in inline-flex items-center gap-2 rounded-full border border-white/15 px-3 py-1 text-xs font-medium text-white/70">
            <span className="size-1.5 rounded-full bg-brand-500" />
            Microscopic medicinal-plant screening
          </p>
          <h1 className="fade-in mt-6 text-[2.75rem] font-extrabold leading-[1.02] tracking-tight [animation-delay:60ms] sm:text-6xl lg:text-7xl">
            {HEADLINE}
          </h1>
          <p className="fade-in mt-6 max-w-xl text-lg leading-relaxed text-white/70 [animation-delay:120ms]">
            HerbaScope reads a micrograph, compares it against real reference material, measures how far it sits from
            everything it knows, and applies written rules to decide. When the evidence is not there, it says so.
          </p>
          <div className="fade-in mt-9 flex flex-wrap gap-3 [animation-delay:180ms]">
            <Link
              href="/analyze"
              className="press group inline-flex items-center gap-2 rounded-full bg-brand-500 px-6 py-3 font-semibold text-ink-900 hover:bg-brand-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
            >
              Screen a sample
              <ArrowRightIcon aria-hidden="true" weight="bold" className="size-4 transition-transform duration-200 group-hover:translate-x-1" />
            </Link>
            <Link
              href="/history"
              className="press inline-flex items-center gap-2 rounded-full border border-white/25 px-6 py-3 font-medium text-white hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
            >
              <ClockCounterClockwiseIcon aria-hidden="true" className="size-4" />
              Analysis history
            </Link>
          </div>
        </div>

        <div className="fade-in relative justify-self-center [animation-delay:240ms]">
          <Mascot variant="hero" priority alt="Herbie, the HerbaScope mascot" className="w-64 sm:w-80 lg:w-[24rem]" />
        </div>
      </div>

      <div className="border-t border-white/10">
        <dl className="mx-auto grid max-w-6xl grid-cols-2 gap-px bg-white/10 px-5 sm:px-8 lg:grid-cols-4">
          {[
            ["97.2%", "test accuracy (70 of 72)"],
            ["200/200", "field photographs rejected"],
            ["~2.8 s", "per analysis, on a laptop CPU"],
            ["0", "cloud calls, ever"],
          ].map(([value, label]) => (
            <div key={label} className="bg-ink-900 px-1 py-6 lg:px-4">
              <dt className="font-mono text-2xl font-semibold tabular-nums text-brand-500 sm:text-3xl">{value}</dt>
              <dd className="mt-1 text-xs text-white/55">{label}</dd>
            </div>
          ))}
        </dl>
      </div>

      <p className="pointer-events-none absolute bottom-4 right-5 hidden items-center gap-2 text-xs text-white/40 lg:flex">
        <ArrowDownIcon aria-hidden="true" className="size-3.5" />
        Scroll
      </p>
    </section>
  );
}

export function Problem() {
  return (
    <Section tone="paper">
      <div className="grid items-center gap-12 lg:grid-cols-2">
        <div>
          <Eyebrow>The problem</Eyebrow>
          <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-5xl">
            A classifier must answer. A screening tool must be able to refuse.
          </h2>
          <p className="mt-6 text-lg leading-relaxed text-muted">
            Show a two-class model a photograph of a garden leaf and it still returns one of its two classes, with a
            confident-looking number attached. We measured it: a classifier on its own accepted{" "}
            <span className="font-semibold text-ink">all 200</span> out-of-distribution field photographs as microscopy.
          </p>
          <p className="mt-4 text-lg leading-relaxed text-muted">
            The useful questions are different. Does this sample resemble known reference material? Do the references
            agree with the prediction? Is the image even good enough to judge?
          </p>
        </div>

        <Reveal className="rounded-3xl border border-line bg-surface p-6 sm:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted">Measured, layer by layer</p>
          <ol className="mt-6 space-y-5">
            {[
              ["Classifier alone", "accepted 200 of 200 field photographs", "fail"],
              ["+ reference agreement", "still accepted 121 of 200", "review"],
              ["+ unknown detection", "accepted 0 of 200", "pass"],
              ["+ decision policy", "19 passes on the test set, 18 correct", "pass"],
            ].map(([layer, result, tone], index) => (
              <li key={layer} className="flex gap-4">
                <span className="mt-1 font-mono text-xs text-muted">{String(index + 1).padStart(2, "0")}</span>
                <div className="min-w-0">
                  <p className="font-semibold text-ink">{layer}</p>
                  <p className={`text-sm ${tone === "pass" ? "text-pass-700" : tone === "review" ? "text-review-700" : "text-fail-600"}`}>
                    {result}
                  </p>
                </div>
              </li>
            ))}
          </ol>
          <p className="mt-6 border-t border-line pt-4 text-xs text-muted">
            From the generated evaluation report: DIMPSAR evaluation split and the Mikrobat test split.
          </p>
        </Reveal>
      </div>
    </Section>
  );
}

const PIPELINE = [
  { icon: SealCheckIcon, title: "Quality", text: "Focus, exposure, clipping and specimen visibility, against bounds calibrated on reference material." },
  { icon: BrainIcon, title: "Representation", text: "A frozen DINOv2 encoder turns four rotations of the micrograph into a single embedding." },
  { icon: TargetIcon, title: "Classifier", text: "Logistic regression estimates which supported class the embedding resembles." },
  { icon: ImagesIcon, title: "Retrieval", text: "FAISS finds the nearest curated reference micrographs and shows them to you." },
  { icon: QuestionIcon, title: "Unknown", text: "Distance to the reference library, against a threshold calibrated on real out-of-distribution data." },
  { icon: ScalesIcon, title: "Evidence", text: "Classifier and retrieval compared as complementary analyses of the same representation." },
  { icon: GavelIcon, title: "Decision", text: "Versioned, deterministic rules. No model and no language model can override the outcome." },
];

export function Pipeline() {
  return (
    <div className="bg-surface text-ink">
      <PinnedRail
        heading={
          <>
            <Eyebrow>How a decision is made</Eyebrow>
            <h2 className="mt-4 max-w-3xl text-3xl font-bold tracking-tight sm:text-5xl">
              Seven stages, and every one of them is shown to you.
            </h2>
            <p className="mt-4 max-w-xl text-muted">Keep scrolling: the stages move as you go.</p>
          </>
        }
      >
        {PIPELINE.map(({ icon: Icon, title, text }, index) => (
          <article
            key={title}
            className="lift h-full w-[19rem] shrink-0 snap-start rounded-3xl border border-line bg-canvas p-6 sm:w-[21rem]"
          >
            <div className="flex items-center justify-between">
              <span className="grid size-11 place-items-center rounded-2xl bg-ink-900 text-brand-500">
                <Icon aria-hidden="true" weight="duotone" className="size-5" />
              </span>
              <span className="font-mono text-xs text-muted">{String(index + 1).padStart(2, "0")}</span>
            </div>
            <h3 className="mt-5 text-lg font-bold">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{text}</p>
          </article>
        ))}
      </PinnedRail>
    </div>
  );
}

const OUTCOMES = [
  {
    icon: CheckCircleIcon,
    title: "Preliminary pass",
    text: "Confident prediction, unanimous reference support, low unknown risk and acceptable image quality.",
    accent: "border-pass-200 bg-pass-50",
    chip: "text-pass-700",
  },
  {
    icon: WarningIcon,
    title: "Review required",
    text: "Evidence is borderline or conflicting, for example the classifier and the reference library disagree.",
    accent: "border-review-200 bg-review-50",
    chip: "text-review-700",
  },
  {
    icon: QuestionIcon,
    title: "Unknown",
    text: "The sample sits outside the calibrated reference distribution, so no screening conclusion is drawn.",
    accent: "border-unknown-200 bg-unknown-50",
    chip: "text-unknown-700",
  },
];

export function Outcomes() {
  return (
    <Section tone="paper">
      <Eyebrow>Three possible answers</Eyebrow>
      <h2 className="mt-4 max-w-3xl text-3xl font-bold tracking-tight sm:text-5xl">
        &ldquo;I don&rsquo;t know&rdquo; is a first-class result.
      </h2>
      <div className="mt-12 grid gap-5 md:grid-cols-3">
        {OUTCOMES.map(({ icon: Icon, title, text, accent, chip }, index) => (
          <Reveal key={title} index={index} className={`lift rounded-3xl border p-6 ${accent}`}>
            <Icon aria-hidden="true" weight="duotone" className={`size-8 ${chip}`} />
            <h3 className="mt-4 text-lg font-bold">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink/75">{text}</p>
          </Reveal>
        ))}
      </div>
      <p className="mt-8 max-w-2xl text-sm text-muted">
        The thresholds behind these outcomes are not typed in by hand. Each one is calibrated against a written
        objective on data the model never trained on, and stored with its provenance.
      </p>
    </Section>
  );
}

export function Evidence() {
  return (
    <Section tone="dark">
      <div className="grid items-center gap-12 lg:grid-cols-2">
        <div>
          <Eyebrow tone="light">Evidence you can inspect</Eyebrow>
          <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-5xl">Not a score. A report you can argue with.</h2>
          <ul className="mt-8 space-y-4 text-white/75">
            {[
              "The five nearest reference micrographs, openable side by side with your sample.",
              "A distance gauge with the calibrated known boundary and unknown threshold marked on it.",
              "Every pass criterion with its observed value, its required value, and whether it was met.",
              "The exact artifact versions that produced the result.",
            ].map((item) => (
              <li key={item} className="flex gap-3">
                <CheckCircleIcon aria-hidden="true" weight="fill" className="mt-0.5 size-5 shrink-0 text-brand-500" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <Reveal className="rounded-3xl border border-white/10 bg-ink-800 p-6 sm:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/50">Screening decision</p>
          <p className="mt-3 inline-flex items-center gap-2 rounded-full bg-brand-500 px-3 py-1 text-sm font-bold text-ink-900">
            <CheckCircleIcon aria-hidden="true" weight="bold" className="size-4" />
            Preliminary pass
          </p>
          <dl className="mt-6 space-y-3 font-mono text-sm">
            {[
              ["classifier confidence", "0.999", "≥ 0.627"],
              ["top reference similarity", "0.972", "≥ 0.897"],
              ["unknown risk", "0.004", "≤ 0.042"],
              ["evidence agreement", "HIGH", "HIGH"],
            ].map(([name, observed, required]) => (
              <div key={name} className="flex items-baseline justify-between gap-4 border-b border-white/10 pb-2">
                <dt className="text-white/60">{name}</dt>
                <dd className="tabular-nums">
                  <span className="text-brand-500">{observed}</span> <span className="text-white/40">{required}</span>
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-xs text-white/50">A real result from the bundled demo sample, not a mock-up.</p>
        </Reveal>
      </div>
    </Section>
  );
}

const REPORT_CARDS = [
  { icon: GavelIcon, title: "Decision", text: "The outcome, the reason it was reached, and every pass criterion with observed and required values." },
  { icon: TargetIcon, title: "Prediction", text: "Class probabilities with the calibrated policy threshold marked on the bar." },
  { icon: ImagesIcon, title: "Reference atlas", text: "The five nearest reference micrographs, each openable beside your sample." },
  { icon: GaugeIcon, title: "Unknown risk", text: "Where the sample falls between the known boundary and the unknown threshold." },
  { icon: SealCheckIcon, title: "Image quality", text: "Six checks with the measured value and the calibrated bound for each." },
  { icon: ScalesIcon, title: "Provenance", text: "Encoder, classifier, index, calibration and policy versions behind this exact result." },
];

export function ReportAnatomy() {
  return (
    <section className="bg-surface text-ink">
      <div className="mx-auto w-full max-w-6xl px-5 pb-6 pt-20 sm:px-8 lg:pt-28">
        <Eyebrow>What you get back</Eyebrow>
        <h2 className="mt-4 max-w-3xl text-3xl font-bold tracking-tight sm:text-5xl">Six panels, one screening report.</h2>
      </div>
      <div className="mx-auto max-w-6xl pb-20 lg:pb-28">
        <SnapRail label="Report panels">
          {REPORT_CARDS.map(({ icon: Icon, title, text }) => (
            <article key={title} className="lift w-[17rem] shrink-0 snap-start rounded-3xl border border-line bg-canvas p-6 sm:w-[19rem]">
              <Icon aria-hidden="true" weight="duotone" className="size-7 text-brand-700" />
              <h3 className="mt-4 font-bold">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{text}</p>
            </article>
          ))}
        </SnapRail>
      </div>
    </section>
  );
}

const SUPPORTED = [
  "Microscopic visual screening",
  "Reference-consistency checks with visible references",
  "Unknown and out-of-distribution rejection",
  "Detection of classifier and reference disagreement",
];

const NOT_CLAIMED = [
  "Definitive authentication or a laboratory replacement",
  "Chemical authentication",
  "Adulteration detection",
  "Clinical diagnosis or regulatory certification",
];

export function Limits() {
  return (
    <Section tone="paper">
      <Eyebrow>Where the claim stops</Eyebrow>
      <h2 className="mt-4 max-w-3xl text-3xl font-bold tracking-tight sm:text-5xl">
        Two supported species, and a limit stated in every result.
      </h2>
      <div className="mt-12 grid gap-5 md:grid-cols-2">
        <Reveal className="rounded-3xl border border-line bg-surface p-6">
          <h3 className="font-bold">What HerbaScope supports</h3>
          <ul className="mt-4 space-y-3 text-sm text-muted">
            {SUPPORTED.map((item) => (
              <li key={item} className="flex gap-2">
                <CheckCircleIcon aria-hidden="true" weight="fill" className="mt-0.5 size-4 shrink-0 text-pass-600" />
                {item}
              </li>
            ))}
          </ul>
        </Reveal>
        <Reveal index={1} className="rounded-3xl border border-line bg-surface p-6">
          <h3 className="font-bold">What it does not claim</h3>
          <ul className="mt-4 space-y-3 text-sm text-muted">
            {NOT_CLAIMED.map((item) => (
              <li key={item} className="flex gap-2">
                <XCircleIcon aria-hidden="true" weight="fill" className="mt-0.5 size-4 shrink-0 text-fail-600" />
                {item}
              </li>
            ))}
          </ul>
        </Reveal>
      </div>
      <p className="mt-8 max-w-3xl text-sm leading-relaxed text-muted">
        Conclusions only transfer to fragment types the reference library contains. On material it has never seen, the
        classifier and the retrieval can agree confidently on the wrong species, because they read the same
        representation. That is measured, published in the evaluation report, and repeated on every result page.
      </p>
    </Section>
  );
}

export function CallToAction() {
  return (
    <Section tone="dark" className="grid-field">
      <div className="grid items-center gap-12 lg:grid-cols-[1fr_22rem]">
        <div>
          <Eyebrow tone="light">Ready when you are</Eyebrow>
          <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-5xl">
            Drop in a micrograph. Everything runs on this machine.
          </h2>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-white/70">
            No cloud service, no language model, no hardware. The encoder, the reference library and the calibrated
            policy all live locally, and the result is stored only here.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-5">
            <Link
              href="/analyze"
              className="press group inline-flex items-center gap-2 rounded-full bg-brand-500 px-6 py-3 font-semibold text-ink-900 hover:bg-brand-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
            >
              Upload a sample
              <ArrowRightIcon aria-hidden="true" weight="bold" className="size-4 transition-transform duration-200 group-hover:translate-x-1" />
            </Link>
            <Mascot variant="spot" className="hidden w-20 sm:block" />
          </div>
        </div>
        <div className="[&_dd]:text-white [&_h2]:text-white [&_section]:border-white/10 [&_section]:bg-ink-800 [&_section]:text-white">
          <SystemStatus />
        </div>
      </div>
    </Section>
  );
}
