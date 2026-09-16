# Security policy

## Scope

HerbaScope is a local-first application: the API, the model artifacts and the stored analyses all
live on the machine that runs it. There is no hosted service, and no sample image or result leaves
the workstation.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/gowtham472/herbascope/security/advisories/new)
rather than a public issue. Please include the version or commit, reproduction steps, and what an
attacker could achieve. Expect an acknowledgement within a week.

## What the application already does

- Uploads are checked against a MIME allow-list, a size cap read without buffering the oversized
  file, and a full decode with a 50-megapixel guard before anything is analysed.
- Uploaded bytes are never stored or served back; samples are re-encoded to PNG.
- Identifiers are server-generated UUIDs with strict patterns, and reference image paths must
  resolve inside the data directory.
- CORS is restricted to configured origins, and containers run as non-root users.
- Model artifacts are located through a manifest and verified by SHA-256 at load, so a modified or
  partially replaced artifact is rejected before it can serve a result.

## Intended use

HerbaScope provides preliminary visual screening support. It does not replace laboratory
confirmation or expert botanical authentication, and it must not be used as the sole basis for a
release, rejection or certification decision.
