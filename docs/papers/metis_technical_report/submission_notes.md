# Submission Notes

## Recommended arXiv Category

Primary: `cs.LG`

Possible cross-list: `cs.CL` if the final version emphasizes language-model
training and evaluation more than systems/architecture.

## Before Upload

- Add benchmark results for released checkpoints when available. At minimum:
  - lightweight generation sanity suite;
  - perplexity or held-out validation loss from a valid run;
  - a comparison between base, chat, and think variants if all are included.
- Add a small architecture diagram or routing schematic.
- Re-check every claimed number against the current manifests.
- Keep Metis-1.4 framed as corrected/current; mention the older bad objective
  only as provenance.

## Suggested Abstract Tone

Keep the abstract sober. The contribution is not "state of the art." The
contribution is a transparent small-model research line that documents:

- dense, recurrent-hybrid, and sparse-MoE architectural iterations;
- operational lessons from training under limited compute;
- the move from released 200M/500M-class checkpoints toward a 1B-class sparse
  expert-parallel design whose current exact manifest is 897.4M params;
- explicit failure accounting and compute gates.

That is more credible than overclaiming benchmark performance.
