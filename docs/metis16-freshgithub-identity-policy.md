# FreshGitHub repository identity policy

FreshGitHub does not treat the minimal `repo` object in GH Archive as proof
that a repository is original. That object normally omits fork and mirror
fields.

The acquisition materializer applies this fail-closed policy:

1. Explicit full repository objects embedded in `ForkEvent` and pull-request
   payloads can reject a known fork or mirror before any source archive is
   downloaded.
2. Every remaining repository must have an explicit GitHub REST repository
   response with `fork: false` and a null `mirror_url` before it is accepted.
3. The normalized decision is cached in
   `cache/github-policy/repository-identity.sqlite3`. Its evidence fields carry
   a SHA-256 integrity digest, observation time, API version, parent, and mirror
   URL.
4. Every emitted code record and associated discussion carries the canonical
   decision and evidence digest. Unknown, unavailable, contradictory, forked,
   or mirrored repositories are never admitted.

Production FreshGitHub acquisition therefore requires a read-only
`GITHUB_TOKEN` or `GH_TOKEN`. The token is sent only in the HTTPS authorization
header and is not written to records, receipts, or the cache.

This proves GitHub's repository classification at the recorded observation. It
does not prove independent authorship, detect a detached fork that GitHub now
classifies as independent, or detect source copied into a new repository. Exact
and structural code deduplication remain responsible for those content-level
cases.
