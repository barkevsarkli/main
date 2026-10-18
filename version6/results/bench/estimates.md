Workers here: 8. AWS: 192 workers, per-run time / 2.25 (this machine vs c8g.48xlarge on the version5 code path).

| stage | variant | runs | epochs | here (h) | target (h) | AWS (h) | AWS cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| stage0 | full | 240 | 9600 | 1.5 | 4 | 0.05 | 0 |
| stage1 | full | 4554 | 182160 | 198.2 | 72 | 4.00 | 10 |
| stage1 | seeds 3 -> 2 | 3036 | 121440 | 132.4 | 72 | 2.77 | 7 |
| stage1 | + drop K = 7 | 2292 | 91680 | 97.1 | 72 | 2.12 | 6 |
| stage1 | + drop lowest LR | 1848 | 73920 | 78.4 | 72 | 1.77 | 5 |
| stage2 | full | 12740 | 509600 | 494.9 | 120 | 9.51 | 25 |
| stage2 | sweep seeds 10 -> 6 | 11172 | 446880 | 434.1 | 120 | 8.38 | 22 |
| stage2 | + extension 30 -> 20 seeds | 8232 | 329280 | 320.0 | 120 | 6.27 | 16 |
| stage2 | + drop @conv/@hidden at H=256 | 7896 | 315840 | 298.1 | 120 | 5.86 | 15 |
| stage2 | + drop sigmoid CIFAR H=256 | 7644 | 305760 | 274.6 | 120 | 5.42 | 14 |
| stage3 | full | 450 | 18000 | 17.9 | 24 | 0.49 | 1 |
