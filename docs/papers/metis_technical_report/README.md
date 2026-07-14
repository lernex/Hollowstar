# Metis Technical Report Draft

This folder contains an arXiv-style technical report draft for the Metis model
line.

## Files

- `main.tex`: paper source.
- `references.bib`: BibTeX mirror for provenance; `main.tex` currently uses an
  inline bibliography to make arXiv source processing simpler.
- `submission_notes.md`: what must be verified or filled in before arXiv upload.

## Build

From this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `pdflatex` is unavailable locally, upload `main.tex` and `references.bib` to
Overleaf or another LaTeX environment.

## Current Status

This is a serious technical-report draft. It deliberately frames Metis-1.5 as a
current architecture and training plan, not as a completed benchmark release.
Metis-1.4 is described as corrected/current, while the older label-shift issue is
kept only as provenance.
