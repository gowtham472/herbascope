import Image from "next/image";

/**
 * The supplied HerbaScope wordmark (leaf pair plus lettering), served as-is from
 * `public/brand/wordmark.svg`.
 */
export function Wordmark({ className = "" }: { className?: string }) {
  return <Image src="/brand/wordmark.svg" width={791} height={260} alt="HerbaScope" unoptimized priority className={className} />;
}
