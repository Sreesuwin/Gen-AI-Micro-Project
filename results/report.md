# Smart Prompt Optimization - Results Report

- Dataset items: **6**
- Total model calls measured: **55**
- Model: `gemma-4-26b-a4b-it`

## Baseline (no compression)

- Mean input tokens: 1140.4
- Mean cost per query: $0.005331
- Mean Token-F1: 0.6153
- Mean judge score: n/a

## All configurations

| method | level | ratio | tokens in | cost/query | F1 | judge |
|---|---|---|---|---|---|---|
| baseline | 1.0 | 1.000 | 1140.4 | $0.005331 | 0.6153 | - |
| lexical | 1.0 | 1.000 | 1140.4 | $0.005331 | 0.6153 | - |
| lexical | 0.9 | 0.808 | 909.3 | $0.003415 | 0.5693 | - |
| lexical | 0.75 | 0.682 | 819.0 | $0.004824 | 0.2533 | - |
| lexical | 0.6 | 0.573 | 646.0 | $0.006963 | 0.2159 | - |
| lexical | 0.45 | 0.489 | 551.5 | $0.006439 | 0.1889 | - |
| lexical | 0.3 | 0.427 | 496.7 | $0.003361 | 0.0336 | - |
| lexical | 0.15 | 0.251 | 283.0 | $0.005396 | 0.1295 | - |
| sentence | 1.0 | 1.000 | 1140.4 | $0.005331 | 0.6153 | - |
| sentence | 0.9 | 0.892 | 1024.0 | $0.002396 | 0.6321 | - |
| sentence | 0.75 | 0.837 | 873.3 | $0.001847 | 0.5855 | - |
| sentence | 0.6 | 0.715 | 695.8 | $0.003686 | 0.6291 | - |
| sentence | 0.45 | 0.653 | 550.7 | $0.002141 | 0.5323 | - |
| sentence | 0.3 | 0.511 | 394.5 | $0.002718 | 0.5887 | - |
| sentence | 0.15 | 0.483 | 259.7 | $0.001600 | 0.6000 | - |

## Optimisation result

- Constraint: mean judge score >= **0.0** and mean F1 >= **90%** of baseline (floor 0.5538)

**Selected configuration: method = `sentence`, level = 0.15**

- Compression ratio achieved: 0.483
- Input tokens: 1140.4 -> 259.7 (**77.23% fewer**)
- Cost per query: $0.005331 -> $0.001600 (**69.99% cheaper**)
- Token-F1 retained: 97.51% of baseline
- Judge score: 0.00 (threshold 0.0)
- Feasible candidates satisfying the constraint: 8

## Charts

- `results/cost_vs_level.png`
- `results/quality_vs_level.png`
- `results/pareto.png`
