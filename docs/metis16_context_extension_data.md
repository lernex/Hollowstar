# Metis-1.6 131K context-extension data

The machine-readable contract is
`manifests/context-extension.yaml`. It adds retrieval reserve to the existing
login2 acquisition plan, then Rhea selects and packs exactly 18,000,000,000
unique active tokens. The continued-pretraining budget is therefore 18B
tokens, not 163,840 tokens: 163,840 is the training sequence length used to
deploy a 131,072-token context window.

## Research result

The selected recipe combines four findings:

- [Data Engineering for Scaling Language Models to 128K
  Context](https://arxiv.org/abs/2402.10171) finds that domain balance and
  within-domain length upsampling both matter; blindly turning the mix into a
  books corpus is inferior.
- [ProLong](https://github.com/princeton-nlp/ProLong) supports repository code,
  books/reference material, educational web, and retained high-quality
  short-context data. Its released 512K dataset is Llama-tokenized, so Metis
  uses its source recipe but never ingests those foreign token IDs.
- [LongFilter](https://arxiv.org/abs/2510.25804) motivates selecting text with
  genuine long-range predictive information rather than raw character length.
  Rhea uses a cheap structural screen over the whole pool; the live Metis
  checkpoint supplies the expensive 4K-versus-131K calibration at the 6B,
  12B, and 18B gates.
- [NExtLong](https://arxiv.org/abs/2501.12766) motivates dependency-preserving
  negative-document extension. Metis constructs these examples only from
  original same-domain chunks and labels them as transformed, never as newly
  generated prose.

No paper establishes 18B as a universal optimum. It is an intentionally
conservative Metis budget, larger than the 1B-5B range demonstrated in the
128K data-engineering study, with autonomous promotion gates so the 18B
checkpoint is not automatically preferred over a better 6B or 12B checkpoint.

## Exact final mix

| Domain | Active tokens | Retrieval candidates |
|---|---:|---:|
| Repository code | 3.6B | 8.235B |
| Science, papers, medicine, patents | 3.6B | 8.420B |
| Books, law, government, long reference | 2.7B | 4.445B |
| Educational and high-quality web | 3.6B | 10.950B |
| Math, proofs, and formal text | 1.8B | 4.520B |
| Technical discussion and current documentation | 1.8B | 5.240B |
| General reference | 0.9B | 2.100B |
| **Total** | **18.0B** | **43.910B** |

The retrieval command already used for the 1T corpus now requests an
additional 25.910B-token long-document reserve from those same 31
license-audited source families. It does not create a second downloader or
duplicate raw corpora. Every source has a fixed same-domain donor order.
Exhausting an entire domain stops the build; generated data and
cross-domain substitution are forbidden.

Rhea selects:

- 70% natural long documents;
- 20% dependency-constructed sequences;
- 10% high-quality 4K replay.

Selection is final-tokenizer measured, document-aware, deterministic, and
two-pass. Every dependency-construction tranche has one dedicated task for
each of the seven domains, and the packer rejects a cross-domain distractor.
Packing emits 110,592 compact 163,840-token rows, exact 6B/12B/18B tranches,
document-start masks, active lengths, and gate IDs. Padding never counts
toward the 18B budget.

## Gate calibration and promotion

Rhea also seals 384 disjoint 131,072-token evaluation records, stratified
across the same seven-domain target mix. Each carries a far-distance
associative probe and a 4,096-token tail. Praxis evaluates three records per
rank and Logos one record per rank, so both family topologies have exactly
balanced work.

Before the first update, the trainer records base-checkpoint short-context,
long-context, and associative-probe metrics. At the first optimizer boundary
after 6B, 12B, and 18B active tokens it:

1. writes and deeply verifies a durable distributed checkpoint;
2. reevaluates all 384 disjoint records;
3. rejects non-finite results, excessive base-loss regression, long-loss
   regression, or a missing first-gate associative gain;
4. seals a score and lineage receipt.

After 18B it restores the highest-scoring passing gate, which may be 6B or
12B. If none passes, SFT never begins.

## Operational path

The login2 acquisition plan is expanded by
`metis_data.manifest.candidate_plan`. Rhea then runs:

1. `context_select`;
2. `context_prepare`;
3. 96 restartable `context_pack` tasks;
4. `context_verify`.

`context_verify` rehashes every array, proves exact active-token and
source/domain accounting, verifies that calibration records are training
disjoint, and emits `metis.context-extension-data/v1`. Portage accepts only
that sealed release.
