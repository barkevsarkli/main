# Summary of mini

81 runs; statuses: {'ok': 81}

## Cells

| dataset | config | hidden | model | lr | n | final_val | final_val_sd | best_val | drop | n_final_lt15 | n_drop_gt10 | n_ever_lt15 | first_collapse_epoch_median | loss_minus_ln10 | test_acc_best | test_f1_best | test_acc | test_f1 | n_not_ok | logit_between_std | dead_frac_conv | dead_frac_hidden1 | own_best_lr | lr_rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist | bank_init/sobel/K3 | 128 | relu | 0.03 | 3 | 97.73 | 0.1617 | 97.73 | 0 | 0 | 0 | 0 |  | -2.238 | 97.67 | 97.65 | 97.67 | 97.65 | 0 | 9.037 | 0 | 0.005208 | 0.03 | 1 |
| mnist | bank_init/sobel/K3 | 128 | relu | 0.1 | 3 | 10.55 | 1.132 | 10.9 | 0.3467 | 3 | 0 | 3 | 1 | 0.02172 | 10.72 | 1.936 | 10.44 | 1.889 | 0 | 3.185e-09 | 1 | 1 | 0.03 | 3.333 |
| mnist | bank_init/sobel/K3 | 128 | relu | 0.3 | 3 | 10.25 | 0.2656 | 10.83 | 0.5867 | 3 | 0 | 3 | 1 | 0.06755 | 10.69 | 1.931 | 9.837 | 1.791 | 0 | 2.023e-08 | 0.9167 | 1 | 0.03 | 10 |
| mnist | bank_init/sobel/K3 | 128 | tanh | 0.03 | 3 | 77.75 | 2.394 | 94.87 | 17.13 | 0 | 3 | 0 |  | -1.794 | 94.54 | 94.42 | 77.3 | 76.6 | 0 | 5.374 |  |  | 0.03 | 1 |
| mnist | bank_init/sobel/K3 | 128 | tanh | 0.1 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 9.303 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 4.235e-07 |  |  | 0.03 | 3.333 |
| mnist | bank_init/sobel/K3 | 128 | tanh | 0.3 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 22.45 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 7.23e-08 |  |  | 0.03 | 10 |
| mnist | bank_init/sobel/K3 | 128 | tanh+relu | 0.03 | 3 | 96.41 | 0.6201 | 96.82 | 0.4067 | 0 | 0 | 0 |  | -2.187 | 96.79 | 96.75 | 96.47 | 96.41 | 0 | 8.629 | 0 | 0.005208 | 0.03 | 1 |
| mnist | bank_init/sobel/K3 | 128 | tanh+relu | 0.1 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 3.559 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 7.394e-08 | 0.75 | 1 | 0.03 | 3.333 |
| mnist | bank_init/sobel/K3 | 128 | tanh+relu | 0.3 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 15.24 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 3.735e-08 | 0.9167 | 1 | 0.03 | 10 |
| mnist | frozen/sobel/K3 | 128 | relu | 0.03 | 3 | 97.7 | 0.14 | 97.71 | 0.006667 | 0 | 0 | 0 |  | -2.247 | 97.67 | 97.65 | 97.6 | 97.58 | 0 | 7.4 | 0 | 0.03646 | 0.03 | 1 |
| mnist | frozen/sobel/K3 | 128 | relu | 0.1 | 3 | 95.9 | 0.4133 | 95.9 | 0 | 0 | 0 | 0 |  | -2.062 | 95.64 | 95.63 | 95.64 | 95.63 | 0 | 15.06 | 0 | 0.3229 | 0.03 | 3.333 |
| mnist | frozen/sobel/K3 | 128 | relu | 0.3 | 3 | 10.25 | 0.2656 | 10.86 | 0.6133 | 3 | 0 | 3 | 1 | 0.06665 | 10.89 | 2.173 | 9.84 | 1.797 | 0 | 5.51e-05 | 0 | 0.9922 | 0.03 | 10 |
| mnist | frozen/sobel/K3 | 128 | tanh | 0.03 | 3 | 97.31 | 0.2139 | 97.38 | 0.07333 | 0 | 0 | 0 |  | -2.234 | 97.21 | 97.18 | 97.18 | 97.15 | 0 | 6.669 |  |  | 0.03 | 1 |
| mnist | frozen/sobel/K3 | 128 | tanh | 0.1 | 3 | 91.71 | 1.232 | 91.71 | 0 | 0 | 0 | 0 |  | -1.282 | 91.31 | 91.15 | 91.31 | 91.15 | 0 | 23.57 |  |  | 0.03 | 3.333 |
| mnist | frozen/sobel/K3 | 128 | tanh | 0.3 | 3 | 84.11 | 0.4965 | 84.11 | 0 | 0 | 0 | 0 |  | 1.536 | 83.62 | 82.87 | 83.62 | 82.87 | 0 | 65.82 |  |  | 0.03 | 10 |
| mnist | frozen/sobel/K3 | 128 | tanh+relu | 0.03 | 3 | 97.95 | 0.2309 | 97.97 | 0.02 | 0 | 0 | 0 |  | -2.25 | 97.87 | 97.84 | 97.86 | 97.83 | 0 | 6.911 | 0 | 0.01562 | 0.03 | 1 |
| mnist | frozen/sobel/K3 | 128 | tanh+relu | 0.1 | 3 | 92.95 | 0.6649 | 92.95 | 0 | 0 | 0 | 0 |  | -1.763 | 92.77 | 92.68 | 92.77 | 92.68 | 0 | 21.61 | 0 | 0.2552 | 0.03 | 3.333 |
| mnist | frozen/sobel/K3 | 128 | tanh+relu | 0.3 | 3 | 80.68 | 2.286 | 80.68 | 0 | 0 | 0 | 0 |  | 1.119 | 80.21 | 79.01 | 80.21 | 79.01 | 0 | 30.4 | 0 | 0.9896 | 0.03 | 10 |
| mnist | random/K3 | 128 | relu | 0.03 | 3 | 97.06 | 0.3219 | 97.47 | 0.4133 | 0 | 0 | 0 |  | -2.234 | 97.32 | 97.29 | 96.92 | 96.87 | 0 | 8.58 | 0.04167 | 0 | 0.03 | 1 |
| mnist | random/K3 | 128 | relu | 0.1 | 3 | 10.55 | 1.132 | 10.9 | 0.3467 | 3 | 0 | 3 | 1 | 0.02172 | 10.72 | 1.936 | 10.44 | 1.889 | 0 | 4.435e-09 | 1 | 1 | 0.03 | 3.333 |
| mnist | random/K3 | 128 | relu | 0.3 | 3 | 10.25 | 0.2656 | 10.83 | 0.5867 | 3 | 0 | 3 | 1 | 0.06755 | 10.69 | 1.931 | 9.837 | 1.791 | 0 | 1.598e-08 | 0.875 | 1 | 0.03 | 10 |
| mnist | random/K3 | 128 | tanh | 0.03 | 3 | 81.91 | 5.11 | 95.39 | 13.49 | 0 | 2 | 0 |  | -1.904 | 95.22 | 95.18 | 81.86 | 81.11 | 0 | 6.182 |  |  | 0.03 | 1 |
| mnist | random/K3 | 128 | tanh | 0.1 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 9.302 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 2.032e-05 |  |  | 0.03 | 3.333 |
| mnist | random/K3 | 128 | tanh | 0.3 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 22.12 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 1.391e-07 |  |  | 0.03 | 10 |
| mnist | random/K3 | 128 | tanh+relu | 0.03 | 3 | 96.41 | 0.9485 | 96.79 | 0.38 | 0 | 0 | 0 |  | -2.199 | 96.59 | 96.54 | 96.43 | 96.38 | 0 | 8.437 | 0 | 0 | 0.03 | 1 |
| mnist | random/K3 | 128 | tanh+relu | 0.1 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 3.558 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 4.764e-07 | 0.4167 | 1 | 0.03 | 3.333 |
| mnist | random/K3 | 128 | tanh+relu | 0.3 | 3 | 10.59 | 1.132 | 10.68 | 0.08667 | 3 | 0 | 3 | 1 | 15.24 | 10.76 | 1.942 | 10.76 | 1.942 | 0 | 1.201e-07 | 0.5 | 1 | 0.03 | 10 |

## Paired comparisons

| dataset | config | hidden | lr | metric | comparison | n | mean_diff | ci_lo | ci_hi | wins | p_t | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist | bank_init/sobel/K3 | 128 | 0.03 | final_val | tanh+relu - relu | 3 | -1.313 | -2.02 | -0.5 | 0 | 0.09706 | 0.6794 |
| mnist | bank_init/sobel/K3 | 128 | 0.03 | test_acc_best | tanh+relu - relu | 3 | -0.8767 | -1.26 | -0.67 | 0 | 0.04471 | 0.313 |
| mnist | bank_init/sobel/K3 | 128 | 0.03 | final_val | tanh+relu - tanh | 3 | 18.67 | 15.58 | 20.96 | 3 | 0.007293 | 0.06564 |
| mnist | bank_init/sobel/K3 | 128 | 0.03 | test_acc_best | tanh+relu - tanh | 3 | 2.253 | 1.87 | 2.5 | 3 | 0.007352 | 0.06617 |
| mnist | bank_init/sobel/K3 | 128 | 0.1 | final_val | tanh+relu - relu | 3 | 0.04 | -0.26 | 0.38 | 1 | 0.8495 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.1 | test_acc_best | tanh+relu - relu | 3 | 0.04 | -0.85 | 0.97 | 1 | 0.9463 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.1 | final_val | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.1 | test_acc_best | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.3 | final_val | tanh+relu - relu | 3 | 0.3467 | -0.26 | 1.46 | 1 | 0.5974 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.3 | test_acc_best | tanh+relu - relu | 3 | 0.06667 | -0.85 | 1.72 | 1 | 0.9432 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.3 | final_val | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | bank_init/sobel/K3 | 128 | 0.3 | test_acc_best | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | frozen/sobel/K3 | 128 | 0.03 | final_val | tanh+relu - relu | 3 | 0.2467 | 0.04 | 0.48 | 3 | 0.1932 | 1 |
| mnist | frozen/sobel/K3 | 128 | 0.03 | test_acc_best | tanh+relu - relu | 3 | 0.2033 | 0.11 | 0.35 | 3 | 0.1115 | 0.6687 |
| mnist | frozen/sobel/K3 | 128 | 0.03 | final_val | tanh+relu - tanh | 3 | 0.64 | 0.26 | 1.02 | 3 | 0.1002 | 0.7012 |
| mnist | frozen/sobel/K3 | 128 | 0.03 | test_acc_best | tanh+relu - tanh | 3 | 0.66 | 0.49 | 0.9 | 3 | 0.03324 | 0.2659 |
| mnist | frozen/sobel/K3 | 128 | 0.1 | final_val | tanh+relu - relu | 3 | -2.947 | -4 | -2.18 | 0 | 0.0325 | 0.26 |
| mnist | frozen/sobel/K3 | 128 | 0.1 | test_acc_best | tanh+relu - relu | 3 | -2.873 | -3.34 | -2.61 | 0 | 0.006566 | 0.05252 |
| mnist | frozen/sobel/K3 | 128 | 0.1 | final_val | tanh+relu - tanh | 3 | 1.247 | -0.26 | 2.52 | 2 | 0.264 | 1 |
| mnist | frozen/sobel/K3 | 128 | 0.1 | test_acc_best | tanh+relu - tanh | 3 | 1.457 | -0.44 | 2.47 | 2 | 0.2646 | 1 |
| mnist | frozen/sobel/K3 | 128 | 0.3 | final_val | tanh+relu - relu | 3 | 70.43 | 67.76 | 72.68 | 3 | 0.0004155 | 0.00374 |
| mnist | frozen/sobel/K3 | 128 | 0.3 | test_acc_best | tanh+relu - relu | 3 | 69.32 | 67.42 | 71.11 | 3 | 0.0002366 | 0.00213 |
| mnist | frozen/sobel/K3 | 128 | 0.3 | final_val | tanh+relu - tanh | 3 | -3.433 | -6.24 | -1.78 | 0 | 0.1354 | 0.8123 |
| mnist | frozen/sobel/K3 | 128 | 0.3 | test_acc_best | tanh+relu - tanh | 3 | -3.413 | -5.72 | -2.11 | 0 | 0.09819 | 0.5892 |
| mnist | random/K3 | 128 | 0.03 | final_val | tanh+relu - relu | 3 | -0.6467 | -1.4 | -0.14 | 0 | 0.2343 | 1 |
| mnist | random/K3 | 128 | 0.03 | test_acc_best | tanh+relu - relu | 3 | -0.73 | -1.48 | 0.03 | 1 | 0.236 | 1 |
| mnist | random/K3 | 128 | 0.03 | final_val | tanh+relu - tanh | 3 | 14.51 | 9.58 | 17.36 | 3 | 0.02787 | 0.2229 |
| mnist | random/K3 | 128 | 0.03 | test_acc_best | tanh+relu - tanh | 3 | 1.367 | 0.79 | 1.76 | 3 | 0.04347 | 0.3043 |
| mnist | random/K3 | 128 | 0.1 | final_val | tanh+relu - relu | 3 | 0.04 | -0.26 | 0.38 | 1 | 0.8495 | 1 |
| mnist | random/K3 | 128 | 0.1 | test_acc_best | tanh+relu - relu | 3 | 0.04 | -0.85 | 0.97 | 1 | 0.9463 | 1 |
| mnist | random/K3 | 128 | 0.1 | final_val | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | random/K3 | 128 | 0.1 | test_acc_best | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | random/K3 | 128 | 0.3 | final_val | tanh+relu - relu | 3 | 0.3467 | -0.26 | 1.46 | 1 | 0.5974 | 1 |
| mnist | random/K3 | 128 | 0.3 | test_acc_best | tanh+relu - relu | 3 | 0.06667 | -0.85 | 1.72 | 1 | 0.9432 | 1 |
| mnist | random/K3 | 128 | 0.3 | final_val | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| mnist | random/K3 | 128 | 0.3 | test_acc_best | tanh+relu - tanh | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
