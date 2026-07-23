# Metis-1.6 site intake and confirmed environment facts

Do not send SSH keys, passwords, Hugging Face token values, GitHub token values, or other secrets.
The pipeline only needs the environment facts below.

## Confirmed login2 acquisition environment

- Account: `vollmerc`; group: `sumusa`.
- Acquisition host: `login2`.
- Acquisition root: `/lus/lustre1/vollmerc/metis-1.6`.
- Python: `/usr/bin/python3.11`.
- GNU Screen, Apptainer, `lfs`, Hugging Face/GitHub/Common Crawl HTTPS access: available.
- A `0/0` quota report is ambiguous; confirm at least 25TB usable capacity before launch. The
  launcher also checks filesystem free space and refuses to start below that floor.
- The repository uses the user's Hugging Face read token. Never send the token by email or commit it.
- The token-owning account must accept the HLE and GPQA gates in addition to the NVIDIA dataset
  gates, because those evaluation-only records are required for fail-closed decontamination.
- First-run bootstrap also needs HTTPS access to PyPI package files, and the account owner needs
  read access to the private `lernex/Metis` GitHub repository.

After cloning the release branch, the account owner runs exactly:

```bash
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

The launcher starts the restart-safe downloader in GNU Screen; SSH may disconnect afterward. Rerun
the same command to resume. For monitoring:

```bash
METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6 ./metisctl status --profile login2
screen -r metis16-acquisition
```

## Still needed for the later Rhea CPU build environment

Ask for:

- the Rhea login/submission host and the exact path by which Rhea sees the acquired data;
- Slurm account/project, CPU partition, QoS, and any reservation name;
- maximum array size, maximum simultaneous jobs, and recommended array concurrency;
- per-job CPU, memory, wall-time, and local-scratch limits;
- required modules for Python 3.11/3.12, Apptainer, compiler/runtime libraries, and Lustre tools;
- whether CPU jobs can read the same files produced by the download server without a transfer step;
- the site-preferred way to request 256GB-memory CPU work for the contamination-index build and
  512GB-memory work for MinHash clustering/selection;
- whether compute nodes have outbound internet access (the build does not require it once acquisition
  and holdouts are complete, but this determines what diagnostics can run there).

## Full environment split

`login2` performs network acquisition onto Lustre. Rhea later performs normalization, quality
gating, exact/repeated-span/near/code deduplication, benchmark decontamination, tokenizer training,
tokenization, exact selection, and sharding. Portage only consumes the immutable verified release
for model pretraining.

Once the still-unknown Rhea values are confirmed:

```bash
export METIS_LUSTRE_ROOT=/exact/path-visible-on-rhea
export METIS_SLURM_ACCOUNT=CONFIRMED
export METIS_SLURM_PARTITION=CONFIRMED
export METIS_SLURM_MAX_ARRAY_SIZE=CONFIRMED
./ops/bootstrap.sh --profile rhea --role compute --lustre-root "$METIS_LUSTRE_ROOT"
./metisctl doctor --profile rhea --role compute
./metisctl submit build --profile rhea
```

The checked-in Rhea profile deliberately refuses to submit until these site values and the separate
data-license review are complete. Do not substitute Portage scheduler settings.
