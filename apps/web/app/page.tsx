import {
  CallToAction,
  Evidence,
  Hero,
  Limits,
  Outcomes,
  Pipeline,
  Problem,
  ReportAnatomy,
} from "@/components/marketing/sections";

/**
 * The marketing page scrolls vertically like any site. Two sections move sideways: the
 * pipeline stages, which are driven by scroll position while the section is pinned, and the
 * report panels, which are a snap rail with arrow controls.
 */
export default function HomePage() {
  return (
    <>
      <Hero />
      <Problem />
      <Pipeline />
      <Outcomes />
      <Evidence />
      <ReportAnatomy />
      <Limits />
      <CallToAction />
    </>
  );
}
