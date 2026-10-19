Workers here: 192. AWS: 192 workers, per-run time / 1.01 (this machine vs c8g.48xlarge on the version5 code path).

| stage | variant | runs | epochs | here (h) | target (h) | AWS (h) | AWS cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| stage0 | full | 240 | 9600 | 0.1 | 4 | 0.06 | 0 |
| stage1 | full | 4554 | 182160 | 4.2 | 72 | 4.15 | 11 |
| stage2 | full | 12740 | 509600 | 10.9 | 120 | 10.74 | 28 |
| stage3 | full | 450 | 18000 | 0.6 | 24 | 0.59 | 2 |
