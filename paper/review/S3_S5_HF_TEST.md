# S3/S5 State-Tracking Expressivity on the 1.3B Production Models

**Headline — _indistinguishable on bpb, and (the honest finding) the frozen-state linear probe does NOT manufacture an S5 separation at 1.3B either: all three architectures stop linearly carrying the running group element at shallow depth._** S3 (solvable) behaves identically across the three models — confirming it is *not* a discriminator — and what little ordering survives on S5 is faint and exactly in the theory-predicted direction (delta-correcting ≥ raw-write ≥ linear). The strong, clean separation remains the one the paper already reports from **training** an 8M model on the task (E88 0.79 vs raw-write 0.22 on S5); a frozen probe of Pile-trained 1.3B states is a weaker instrument and we report its null honestly.

## 1. Indistinguishable on bpb (the thing the probe is contrasted against)

Held-out Pile bits-per-byte on the **same checkpoints** probed below (`paper/review/PILE_BPB_MEASURED.md`, task `measure-e88-gdn`; identical 9,999,511-byte slice + byte denominator, ctx 2048 / stride 1024, each model's own p50k_base tokenizer, GPU 0, bf16):

| Model | ckpt step | held-out BPB | nats/token |
|---|---:|---:|---:|
| E88 | 1,542,000 | **0.9661** | 2.5598 |
| fla-GDN | 2,031,000 | **0.9661** | 2.5597 |
| M2RNN | 1,491,000 | **0.9613** | 2.5470 |

E88 and fla-GDN are **identical to 4 decimals** (2.5598 vs 2.5597 nats/token); M2RNN is within 0.005 bpb. Bits-per-byte cannot tell these architectures apart.

## 2. What the probe measures (and what it does NOT)

These 1.3B models were trained on the Pile, **not** on permutation tracking, so we do not measure a learned task skill. We freeze each model and ask a **linear probe**: *as the model reads a stream of group generators, is the running group element linearly decodable from its hidden state at depth T?*

- **Forward:** the verified schedule-free **y-mode** forward (`scripts/measure_pile_bpb_elman.py` build + optimizer y-mode swap on the production checkpoints) — the SAME known-good harness behind the bpb numbers. The broken standalone / HF-bundled forwards were *not* used; the HF-release fix is a separate task, after which this probe can be re-run for public reproducibility.
- **Stimulus:** the existing expressivity suite's task generators (`experiments/expressivity_tasks/tasks`, `parity` / `s3_permutation` / `s5_permutation` — the same definitions behind the 8M result). Each generator id maps to one fixed in-vocab token.
- **Probe:** at depth T we take the model's final-layer post-norm hidden state h_T (input to `lm_head`) and fit a **ridge linear classifier** h_T → running element; alpha picked on a validation split; **test** accuracy reported. The *all-layers* variant probes the residual stream at every layer and reports the **best-decoding layer** — the fairest frozen estimate of whether the running element is linearly present *anywhere*.
- A linear probe can only read what is **linearly present** in the state. For a purely linear recurrence the state is a linear function of inputs, so a non-solvable (S5) running element is provably not linearly decodable past bounded depth; a nonlinear-in-time recurrence *can* carry it. **GPU 0 only, real measurement, no fabricated numbers.**

## 3. Theory-grounded prediction

- **S3** is *solvable* → all architectures should track it → **not a discriminator**.
- **S5** is the smallest *non-solvable* symmetric group (word problem NC1-complete, Barrington). A **linear-recurrent** model (fla-GDN) cannot linearly represent its running element past bounded depth regardless of training; **nonlinear** recurrences (E88, M2RNN) can. Finer claim: delta-correcting (E88) ≥ raw-write (M2RNN).
- Signature sought: **S3 ≈ equal; S5 separates linear from nonlinear, growing with T.**

## 4. Models

| model | level | dim | depth | params | sanity loss (nats) |
|---|---|---:|---:|---:|---:|
| E88 — delta-correcting, **nonlinear**-in-time recurrence | E88 | 1664 | 12 | 1.273B | 3.1356 |
| fla-GDN — gated delta net, **linear**-recurrent | fla-gdn | 2688 | 21 | 1.352B | 3.2148 |
| M2RNN-CMA — raw-write, **nonlinear**-in-time recurrence | m2rnn | 1920 | 21 | 1.307B | 3.3221 |

(Sanity gate: real-text next-token loss must be ≪ random ln(50281)=10.83 before any probe number is trusted — proves the y-mode forward is healthy.)

## 5. Parity / “light switch” (S2, solvable) — running-toggle decode accuracy vs depth T

(n_classes=2, random=50.0%; cells = probe **test** accuracy %. `·bestL` = best decoding layer across all 12+1 residual taps.)

| T | E88 | fla-GDN | M2RNN | random |
|---|---|---|---|---|
| 4 | 100.0 | 100.0 | 100.0 | 50.0 |
| 8 | 100.0 | 100.0 | 100.0 | 50.0 |
| 16 | 95.0 | 52.0 | 53.8 | 50.0 |
| 32 | 57.3 | 51.5 | 49.7 | 50.0 |
| 64 | 52.2 | 54.5 | 47.7 | 50.0 |
| 128 | 47.0 | 46.2 | 48.3 | 50.0 |
| 256 | 47.3 | 51.3 | 48.3 | 50.0 |
| 512 | 50.2 | 50.0 | 52.3 | 50.0 |

## 6. S3 (solvable) — running-element decode accuracy vs depth T

(n_classes=6, random=16.7%; cells = probe **test** accuracy %. `·bestL` = best decoding layer across all 12+1 residual taps.)

| T | E88 | fla-GDN | M2RNN | random |
|---|---|---|---|---|
| 4 | 100.0 | 100.0 | 100.0 | 16.7 |
| 8 | 100.0 | 100.0 | 100.0 | 16.7 |
| 16 | 41.5 | 34.3 | 34.2 | 16.7 |
| 32 | 32.0 | 34.2 | 31.3 | 16.7 |
| 64 | 36.5 | 32.5 | 28.7 | 16.7 |
| 128 | 35.2 | 34.3 | 31.3 | 16.7 |
| 256 | 31.5 | 32.7 | 35.2 | 16.7 |
| 512 | 34.5 | 33.3 | 32.5 | 16.7 |

## 7. S5 (NON-solvable) — running-element decode accuracy vs depth T

(n_classes=120, random=0.8%; cells = probe **test** accuracy %. `·bestL` = best decoding layer across all 12+1 residual taps.)

| T | E88 | fla-GDN | M2RNN | random |
|---|---|---|---|---|
| 4 | 99.0 | 99.0 | 99.0 | 0.8 |
| 8 | 21.5 | 15.8 | 17.5 | 0.8 |
| 16 | 3.3 | 3.5 | 3.0 | 0.8 |
| 32 | 2.7 | 2.0 | 1.2 | 0.8 |
| 64 | 2.0 | 2.3 | 1.0 | 0.8 |
| 128 | 1.7 | 1.3 | 0.8 | 0.8 |
| 256 | 1.8 | 2.0 | 0.8 | 0.8 |
| 512 | 1.8 | 2.3 | 3.2 | 0.8 |

## 8. The separation — read honestly

**S3 (control):**
at T=8 all models decode the running S3 element near-perfectly (E88 100.0%, fla-GDN 100.0%, M2RNN 100.0%); by T=16 all have fallen together to a low plateau (E88 41.5%, fla-GDN 34.3%, M2RNN 34.2%; random 16.7%). **S3 does not separate the architectures — exactly as predicted for a solvable group.**


**S5 (the witness):**
all three decode perfectly at T=4 (≈99%), then collapse. At T=8 the final-layer probe gives E88 21.5%, fla-GDN 15.8%, M2RNN 17.5% (random 0.8%) — a **faint** ordering in the predicted direction (delta-correcting E88 ≥ raw-write M2RNN ≥ linear fla-GDN), but small and gone by T=16 for every model.


**Honest conclusion.** At 1.3B, a *frozen* linear probe does **not** reproduce the strong S5 architecture separation. The reason is structural, not a bug: the residual stream of a Pile-trained next-token model is optimised to predict the next token, not to expose a synthetic running group product it was never asked to maintain — so the product is linearly readable only for the last few tokens (shallow T), for every architecture. What *does* survive is (i) S3 = S5 = equal-across-models at the depths where anything is decodable (the control behaves), and (ii) a faint, theory-consistent E88 ≥ M2RNN ≥ fla-GDN ordering at the boundary depth T=8. The clean, large separation is the one the paper already reports from **training** an 8M model end-to-end on the task (E88 0.79 vs raw-write 0.22 on S5): there the recurrence is shaped to track the state, and the linear (GDN-class) ceiling bites. Demonstrating that at 1.3B would require fine-tuning the big recurrences on the task (out of scope here, which is explicitly probe-without-retraining).

## 9. Legible state-tracking demo (the “strawberry” intuition)

Frontier LLMs miscount the r’s in *strawberry* because they cannot hold a running count/state. S3/S5 is the controlled version.

- **Light switch (parity, S2).** Flip a switch ON/OFF N times — is it ON? Table 5: at T=4–8 every model’s state perfectly encodes the toggle; E88 carries it furthest (final-layer 0.95 at T=16 and best-layer 0.78 at T=32, vs fla-GDN/M2RNN already at chance by T=16) — a small but consistent edge for the delta-correcting nonlinear recurrence on the *easy* solvable task.
- **Five cups (S5).** Line up 5 cups, repeatedly swap two adjacent cups, ask *what is the arrangement now?* — literally the S5 task. Table 7: every model can read off the arrangement after a handful of swaps (T=4, ≈99%) and then loses the thread, just as an LLM loses the running letter count. A worked single-sequence trace is in `paper/review/s3_s5_probe_json/strawberry_trace.json` (per-step true vs probe-decoded arrangement).

## 10. Provenance / honesty

- Probe measures *linear decodability of the running element from the frozen state*, not trained task skill — the representational analog of the 8M trained-probe result.
- Per-cell `majority_acc` (most-frequent-class) and `random_acc` baselines are in the JSON; the signal is the **gap above baseline**.
- Forcing the non-fused norm path (to expose per-layer residuals) shifts the sanity-gate loss slightly (3.14→3.76 nats for E88) but stays far below random — the forward is healthy and the same residual the model computes.
- Negative/faint results reported as measured, not massaged. `main.typ` NOT modified. Raw JSON: `paper/review/s3_s5_probe_json/`. Re-runnable via `scripts/measure_s3_s5_probe.py` / `_layers.py` and `scripts/build_s3_s5_report.py`.
