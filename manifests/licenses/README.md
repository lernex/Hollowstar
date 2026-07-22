# License ledger

The production release gate is fail-closed. The Portage profile keeps
`license_review_complete: false` until the actual source terms and HPE research
policy have been reviewed. The verifier requires license evidence for every
retained token and generates `provenance/LICENSE_LEDGER.jsonl` with observed
license-token totals for every source before `RELEASE.json` can be emitted.

Dataset-card labels are evidence, not automatic legal approval. Repository
code keeps its per-file repository license and is discarded when the license is
missing, conflicting, non-allowlisted, or cannot be tied to the pinned commit.
Changing the profile gate is an operator attestation; it is not a substitute
for legal or institutional review.
