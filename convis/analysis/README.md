# Analysis scripts

Scripts behind the Analysis section of the main README. They read the outputs produced by
`experiments/` and write their results to `results/`.

| Script | Section | Needs model inference |
|---|---|---|
| `caption_agreement.py` | §3 candidate caption agreement | no |
| `category_stats.py` | §4 hallucinated categories | no |
| `extract_flagged.py` | §5 CHAIR false positives | no |
| `consensus_filter.py` | §7 consensus filter | no |
| `rank_analysis.py` | §2 where the intervention happens | yes |

§1 (α sensitivity) needs no separate script — vary `alpha` in the config and rerun
`04convis_decode.py` and `05_eval_chair.py`.

§6 (caption-diversity control) needs no separate script either — set caption generation to
greedy (`do_sample=False` in `02generate_captions.py`) so all n candidates are identical,
then run the pipeline unchanged. The reconstructions still differ because the diffusion
sampler advances between calls.

Each docstring records the expected values, so a reimplementation can be checked against them.
