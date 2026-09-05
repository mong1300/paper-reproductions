# ConVis — Reproduction & Critical Analysis

## Paper Info

**ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models**
Yeji Park, Deokyeong Lee, Junsuk Choe, Buru Chang — AAAI 2025

- arXiv: [2408.13906](https://arxiv.org/abs/2408.13906)
- Official code: [yejipark-m/ConVis](https://github.com/yejipark-m/ConVis)

## Summary

<!-- TODO -->

## Method

<!-- TODO: 3-step pipeline description -->

$$\hat{f} = \frac{1}{n} \sum_{i=1}^{n} \Big( (1 + \alpha) f(v) - \alpha \textstyle f(v'_i)\Big)$$

<!-- TODO: APC (lambda), author settings alpha=1 (captioning) / 0.1 (VQA) -->

## Scope of Reproduction

| Component | Paper | This work |
|---|---|---|
| Backbone | LLaVA-1.5-7B | llava-interleave-qwen-0.5b |
| T2I | Hyper-SDXL | sd-turbo |
| Images | 500 × 3 sets | 100 × 1 set |
| Benchmarks | CHAIR, POPE, MME, HallusionBench, LLaVA-Bench | CHAIR |
| Precision | fp16 | fp16 (MPS) |

<!-- TODO: implications of each substitution; ref. paper Table 6 (CLIPScore vs CHAIR) -->

## Reproduction Results

n=4, λ=0.1, fixed seed(42). Deterministic across reruns (METEOR identical to full precision).

| Setting | CHAIR_S ↓ | CHAIR_I ↓ | METEOR ↑ | Hallucinations |
|---|---|---|---|---|
| Greedy (α=0) | 19.0 | 8.4 | 17.9 | 25 |
| ConVis (α=1) | 16.0 | 6.8 | 18.2 | 21 |
| *Paper, Greedy (7B)* | *22.4* | *7.4* | — | — |
| *Paper, ConVis (7B)* | *18.4* | *6.4* | — | — |

<!-- TODO -->

## Analysis

### 1. α sensitivity

| α | CHAIR_S | CHAIR_I | METEOR | Halluc. | Objects mentioned |
|---|---|---|---|---|---|
| −1 | 14.0 | 5.6 | 17.1 | 16 | 286 |
| −0.8 | 16.0 | 6.0 | 17.0 | 18 | 300 |
| −0.5 | 19.0 | 7.5 | 17.1 | 22 | 293 |
| −0.1 | 17.0 | 7.8 | 17.5 | 24 | 308 |
| 0 | 19.0 | 8.4 | 17.9 | 25 | 298 |
| 0.1 | 17.0 | 6.3 | 17.8 | 19 | 302 |
| 0.5 | 14.0 | 6.2 | 17.9 | 19 | 306 |
| 1 | 16.0 | 6.8 | 18.2 | 21 | 309 |

<!-- TODO: α=−1 → f(v) coefficient 0 (mean of reconstructions only)
           α=−0.8 → all coefficients 1/5 (uniform average, no contrast)
           CAVEAT: differences may be explained by FP rate variation (§6) -->

### 2. Where the intervention happens

Rank of emitted token under f(v), by ambiguity (top1−top2 probability gap).

| gap | α=−1 | α=−0.8 | α=−0.5 | α=−0.1 | α=0.1 | α=0.5 | α=1 |
|---|---|---|---|---|---|---|---|
| < 0.1 | 52.0% | 46.2% | 39.0% | 13.1% | 12.7% | 35.9% | 47.8% |
| 0.1–0.3 | 26.8% | 22.0% | 14.3% | 0.3% | 0.5% | 11.9% | 29.5% |
| 0.3–0.6 | 11.4% | 8.6% | 2.8% | 0.0% | 0.1% | 3.6% | 10.4% |
| > 0.6 | 0.7% | 0.4% | 0.0% | 0.0% | 0.0% | 0.2% | 0.6% |

Rank taken when flipped (α=1, most ambiguous bin): 58% / 21% / 21% (2nd / 3rd / 4th+).

Flip rate is near-symmetric in |α| (row 1 above), while the corrected CHAIR_I of §1 is not:

| \|α\| | flip rate, gap<0.1 (neg / pos) | corrected CHAIR_I (neg / pos) |
|---|---|---|
| 0.1 | 13.1% / 12.7% | 4.9 / 4.0 |
| 0.5 | 39.0% / 35.9% | 4.1 / 3.3 |
| 1 | 52.0% / 47.8% | 3.9 / 3.2 |

<!-- TODO -->

### 3. Candidate caption agreement

COCO objects mentioned across the 4 sampled captions.

| Agreement | Count | Survival in final caption | Halluc. rate among survivors |
|---|---|---|---|
| 4/4 | 143 | 81% | 1.7% |
| 3/4 | 67 | 40% | 11.1% |
| 2/4 | 59 | 17% | 20.0% |
| 1/4 | 133 | 9% | 50.0% |

48% of mentioned objects appeared in only one of four captions.
Survival rates differed by ≤5%p across α, with inconsistent direction.

<!-- TODO -->

### 4. Hallucinated categories

Greedy (α=0), 25 hallucinations across 12 categories.

| Category | Halluc. | Mentioned | Rate | COCO base rate |
|---|---|---|---|---|
| orange | 4 | 4 | 100% | 1.4% |
| knife | 4 | 4 | 100% | 3.5% |
| dining table | 4 | 27 | 15% | 9.9% |
| tv | 3 | 5 | 60% | 3.9% |
| chair | 2 | 3 | 67% | 11.0% |
| bed | 2 | 5 | 40% | 3.2% |

Top-10 most frequent COCO categories account for 28% of hallucinations.
`person` (most frequent category): 0 hallucinations.

<!-- TODO -->

### 5. CHAIR false positives

| Type | Example |
|---|---|
| Color adjective | "a vibrant **orange** coffee cup" |
| Homonym | "a white train **car**", "a sturdy **bed**" (truck bed), "**bears** the inscriptions" |
| Proper noun | "a street scene in **Turkey**" → mapped to bird |
| Synonym over-expansion | `seat`→chair, `container`→bowl, `television`→tv, `houseboat`→boat |
| Part–whole | GT contains `laptop`, caption mentions "its **keyboard**" |
| Depicted object | "a blue shirt adorned with a white **dog**" |

LLM-assisted review of all 164 flagged instances (`results/fp_review.json`):
Claude Opus 5 (high reasoning effort) classified each instance against a six-type rubric;
the file records the judgment, the supporting caption span, and a confidence level for each in korean.

| Category | Count |
|---|---|
| A1 color adjective (`orange`) | 39 |
| A4 synonym over-expansion (`seat`→chair, `container`→bowl, `television`→tv) | 28 |
| A2 homonym (`bow`, `ball`, `bear`, `car`, `bed`) | 18 |
| A6 depicted object | 6 |
| A5 part–whole | 4 |
| A3 proper noun (`Turkey`) | 4 |
| B1 annotation gap | 18 |
| C1 confirmed hallucination | 22 |
| D1 undecidable | 25 |

CHAIR scores under three criteria:

| α | reported S / I | conservative S / I | strict (C1 only) S / I |
|---|---|---|---|
| −1 | 14.0 / 5.6 | 9.0 / 3.9 | 4.0 / 1.4 |
| −0.8 | 16.0 / 6.0 | 9.0 / 3.4 | 3.0 / 1.0 |
| −0.5 | 19.0 / 7.5 | 9.0 / 4.1 | 1.0 / 0.3 |
| −0.1 | 17.0 / 7.8 | 9.0 / 4.9 | 3.0 / 1.6 |
| 0 | 19.0 / 8.4 | 11.0 / 5.7 | 3.0 / 1.4 |
| 0.1 | 17.0 / 6.3 | 10.0 / 4.0 | 3.0 / 1.3 |
| 0.5 | 14.0 / 6.2 | 7.0 / 3.3 | 0.0 / 0.0 |
| 1 | 16.0 / 6.8 | 6.0 / 3.2 | 1.0 / 0.3 |

*conservative*: removes only unambiguous lexical errors (A1, A2, A3, A6 — 67 instances).
*strict*: counts only C1.

<!-- TODO: note that D1 (25) exceeds C1 (22); report bounds, not point estimates -->

### 6. Seed-variant control

Does the contrast signal come from *caption* variation or from *rendering* variation?
One fixed caption (`captions[0]`) was rendered with four different diffusion seeds, keeping
n=4 and α=1 unchanged.

| Setting | CHAIR_S ↓ | CHAIR_I ↓ | METEOR | Hallucinations |
|---|---|---|---|---|
| Greedy (α=0) | 19.0 | 8.4 | 17.9 | 25 |
| ConVis, 4 sampled captions (α=1) | 16.0 | 6.8 | 18.2 | 21 |
| ConVis, 1 caption × 4 seeds (α=1) | 20.0 | 8.1 | 17.3 | 25 |

<!-- TODO -->

### 7. Consensus filter

Objects mentioned in fewer than k of the four candidate captions are removed post-hoc from the
greedy (α=0) caption. No T2I generation, no contrastive decoding.

| Filter | CHAIR_S ↓ | CHAIR_I ↓ | Halluc. | Mentions | Removed | Removed that were hallucinations |
|---|---|---|---|---|---|---|
| none | 19.0 | 8.4 | 25 | 296 | 0 | — |
| k ≥ 1 | 13.0 | 5.2 | 15 | 286 | 10 | 10 (100%) |
| k ≥ 2 | 7.0 | 3.3 | 9 | 274 | 22 | 16 (73%) |
| k ≥ 3 | 5.0 | 2.7 | 7 | 264 | 32 | 18 (56%) |
| k = 4 | 2.0 | 0.9 | 2 | 229 | 67 | 23 (34%) |

This is a post-hoc filter, not a decoding method: it is not subject to fluency constraints, and
CHAIR counts only object mentions, so deletion improves the metric mechanically. Read it as an
upper bound on the information carried by inter-caption agreement, not as a comparable system.

<!-- TODO -->

## Critical Analysis



<!-- TODO
  - α sweep absent (paper reports only 1 and 0.1)
  - side effects unreported (7.4% break rate)
  - effect size vs paper's own std (18.4±0.53, mPLUG-Owl2 17.6±3.54 overlaps OPERA 18.2±0.40)
  - mechanism framing: T2I renders the caption indiscriminately; selectivity originates in f(v),
    v' acts as a uniform probe → implication for Table 6 (T2I quality) and follow-up work (SDCD, ONLY)
  - Figure 5: function words vs hallucinated noun is not a fair comparison; n=1, no statistics
  - reproducibility: generated captions not released → FP rate in paper unverifiable
  - compute cost: n+1 forwards per token, not reported
-->

## Unresolved

<!-- TODO
  - whether the contrast direction selectively targets hallucinated tokens
  - the α < −0.5 regime
-->

## Future Work

If ConVis's effect originates in candidate-caption diversity, the open problem is **how to inject
information from multiple captions into the decoding step**. The authors route this through T2I
generation — a lossy detour that converts captions into images and back into logits. Fixing the
caption and varying only the diffusion seed removed the effect entirely (§6), which suggests the
reconstructed images are a carrier rather than a source of the signal.

A direct route: contrasting against the candidate captions as text, or aggregating their
agreement at the object level is the simplest next step. However, token alignment across captions
of differing length is the main obstacle.

## Repository Structure

```
convis/
├── experiments/                # reproduction pipeline
│   ├── 01prepare_data.py
│   ├── 02generate_captions.py
│   ├── 03generate_images.py
│   ├── 04convis_decode.py
│   ├── 05_eval_chair.py
│   ├── run_all.py
│   └── configs/
├── src/
│   ├── model.py                # convis_decode — Eq. 2
│   ├── t2i.py
│   └── utils.py
├── analysis/                   # TODO
├── eval/                       # CHAIR implementation (from official repo)
└── results/
```

## How to Run

```bash
CONVIS_CONFIG=local_debug.yaml python experiments/run_all.py
```

| Config key | Description |
|---|---|
| `model_id` | MLLM backbone |
| `t2i_id` | T2I model |
| `num_images` | evaluation set size |
| `caption_num` | number of candidate captions (paper: n=4) |
| `alpha` | contrast strength (paper: 1 for captioning) |
| `lam` | adaptive plausibility constraint (paper: 0.1) |
| `device` | `mps` / `cuda` |
| `load_in_4bit` | CUDA only; ignored on MPS |

## Implementation Notes

| Change | Reason |
|---|---|
| `pattern` → `inflect` | `pattern` does not run on Python 3.12 |
| SPICE disabled | Java-based SPICE does not run on Apple Silicon; unrelated to CHAIR |
| Compel added | captions exceed the CLIP 77-token limit |
| Output dirs created | `outputs/coco` missing on a fresh clone |

