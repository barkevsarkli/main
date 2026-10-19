# Summary of stage3

450 runs; statuses: {'ok': 450}

## Cells

| dataset | config | hidden | model | lr | n | final_val | final_val_sd | best_val | drop | n_final_lt15 | n_drop_gt10 | n_ever_lt15 | first_collapse_epoch_median | loss_minus_ln10 | test_acc_best | test_f1_best | test_acc | test_f1 | n_not_ok | logit_between_std | dead_frac_conv | dead_frac_hidden1 | own_best_lr | lr_rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | bank_init/laplacian/K9 | 128 | relu | 0.01 | 15 | 32.02 | 14.16 | 48.6 | 16.58 | 4 | 7 | 6 | 37 | -0.8215 | 48.33 | 47.76 | 31.98 | 30.31 | 0 | 7.276 | 0 | 0.2849 | 0.01 | 1 |
| cifar10 | bank_init/laplacian/K9 | 128 | relu | 0.02 | 15 | 9.929 | 0.4856 | 36.43 | 26.5 | 15 | 15 | 15 | 5 | -0.01759 | 36.11 | 35.2 | 10 | 1.819 | 0 | 2.808e-09 | 0.01667 | 1 | 0.01 | 2 |
| cifar10 | bank_init/laplacian/K9 | 128 | relu | 0.03 | 15 | 10.95 | 3.856 | 25.52 | 14.57 | 14 | 13 | 15 | 2 | -0.008214 | 25.7 | 22.59 | 11.08 | 3.209 | 0 | 0.07925 | 0.1667 | 0.9401 | 0.01 | 3 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh | 0.01 | 15 | 49.25 | 2.2 | 50.78 | 1.531 | 0 | 0 | 0 |  | -0.9954 | 50.84 | 50.08 | 49.66 | 49.04 | 0 | 2.813 |  |  | 0.01 | 1 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh | 0.02 | 15 | 30.75 | 3.384 | 33.72 | 2.971 | 0 | 0 | 3 | 5 | 0.01327 | 33.54 | 30.17 | 31.33 | 27.93 | 0 | 2.078 |  |  | 0.01 | 2 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh | 0.03 | 15 | 18.99 | 5.897 | 23.32 | 4.329 | 5 | 2 | 15 | 1 | 0.9884 | 23.21 | 16.25 | 19.28 | 12.09 | 0 | 1.511 |  |  | 0.01 | 3 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu | 0.01 | 15 | 50.36 | 1.094 | 52.15 | 1.787 | 0 | 0 | 0 |  | -1.531 | 51.52 | 51.11 | 50.51 | 50.21 | 0 | 5.554 | 0 | 0.01458 | 0.01 | 1 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu | 0.02 | 15 | 33.59 | 10.77 | 38.74 | 5.155 | 2 | 2 | 7 | 8 | -0.4043 | 38.63 | 36.76 | 33.63 | 30.74 | 0 | 1.951 | 0.01667 | 0.8562 | 0.01 | 2 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu | 0.03 | 15 | 22.85 | 11.26 | 29.72 | 6.875 | 6 | 4 | 9 | 1 | 0.09609 | 29.52 | 25.41 | 23.03 | 17.83 | 0 | 1.209 | 0.3333 | 0.9698 | 0.01 | 3 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.25 | 0.01 | 15 | 50.85 | 0.971 | 51.92 | 1.071 | 0 | 0 | 0 |  | -1.355 | 51.39 | 50.92 | 50.88 | 50.46 | 0 | 4.439 | 0 | 0.02708 | 0.01 | 1 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.25 | 0.02 | 15 | 33.38 | 3.207 | 35.94 | 2.557 | 0 | 0 | 6 | 10.5 | -0.2589 | 35.76 | 34.01 | 33.56 | 30.99 | 0 | 2.016 | 0.03333 | 0.9167 | 0.01 | 2 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.25 | 0.03 | 15 | 23.79 | 6.079 | 28.25 | 4.461 | 2 | 1 | 11 | 1 | 0.3494 | 28.3 | 23.73 | 24.28 | 18.98 | 0 | 1.689 | 0.5 | 0.9896 | 0.01 | 3 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.75 | 0.01 | 15 | 49.77 | 0.9552 | 51.88 | 2.119 | 0 | 0 | 0 |  | -1.589 | 51.36 | 51.08 | 50.08 | 49.98 | 0 | 6.649 | 0 | 0.01806 | 0.01 | 1 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.75 | 0.02 | 15 | 28.39 | 13.06 | 38.16 | 9.764 | 4 | 5 | 12 | 13.5 | -0.3784 | 38.08 | 36.66 | 28.74 | 25.23 | 0 | 1.431 | 0.02222 | 0.8389 | 0.01 | 2 |
| cifar10 | bank_init/laplacian/K9 | 128 | tanh+relu@0.75 | 0.03 | 15 | 18.18 | 9.435 | 26.08 | 7.895 | 8 | 5 | 14 | 1.5 | 0.00404 | 25.8 | 22.14 | 18.37 | 12.77 | 0 | 0.6649 | 0.2111 | 0.9951 | 0.01 | 3 |
| mnist | bank_init/sobel/K9 | 128 | relu | 0.01 | 15 | 98.19 | 0.3702 | 98.79 | 0.5973 | 0 | 0 | 0 |  | -2.268 | 98.68 | 98.67 | 98.29 | 98.27 | 0 | 31.1 | 0 | 0 | 0.01 | 1 |
| mnist | bank_init/sobel/K9 | 128 | relu | 0.02 | 15 | 10.49 | 0.9024 | 97.88 | 87.39 | 15 | 15 | 15 | 23 | 0.003295 | 97.83 | 97.81 | 10.44 | 1.909 | 0 | 0.02542 | 0.008333 | 0.9995 | 0.01 | 2 |
| mnist | bank_init/sobel/K9 | 128 | relu | 0.03 | 15 | 10.19 | 0.7393 | 95.88 | 85.69 | 15 | 15 | 15 | 8 | 0.005494 | 95.94 | 95.92 | 10.18 | 1.864 | 0 | 0.3741 | 0.08333 | 0.9995 | 0.01 | 3 |
| mnist | bank_init/sobel/K9 | 128 | tanh | 0.01 | 15 | 98.98 | 0.08137 | 99.02 | 0.04 | 0 | 0 | 0 |  | -2.302 | 98.89 | 98.88 | 98.92 | 98.91 | 0 | 8.833 |  |  | 0.01 | 1 |
| mnist | bank_init/sobel/K9 | 128 | tanh | 0.02 | 15 | 98.69 | 0.1035 | 98.74 | 0.04133 | 0 | 0 | 0 |  | -2.303 | 98.5 | 98.48 | 98.61 | 98.6 | 0 | 12.6 |  |  | 0.01 | 2 |
| mnist | bank_init/sobel/K9 | 128 | tanh | 0.03 | 15 | 96.04 | 0.6977 | 96.81 | 0.7693 | 0 | 0 | 0 |  | -2.153 | 96.51 | 96.48 | 95.97 | 95.94 | 0 | 11.34 |  |  | 0.01 | 3 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu | 0.01 | 15 | 98.94 | 0.1193 | 98.98 | 0.044 | 0 | 0 | 0 |  | -2.303 | 98.87 | 98.87 | 98.9 | 98.89 | 0 | 11.86 | 0 | 0.002083 | 0.01 | 1 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu | 0.02 | 15 | 98.29 | 0.3023 | 98.64 | 0.3493 | 0 | 0 | 0 |  | -2.287 | 98.41 | 98.4 | 98.21 | 98.19 | 0 | 22.63 | 0 | 0 | 0.01 | 2 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu | 0.03 | 15 | 97.17 | 0.4095 | 97.67 | 0.5 | 0 | 0 | 0 |  | -2.235 | 97.42 | 97.39 | 97.15 | 97.12 | 0 | 16.7 | 0 | 0.2573 | 0.01 | 3 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.25 | 0.01 | 15 | 99 | 0.07414 | 99.05 | 0.04267 | 0 | 0 | 0 |  | -2.303 | 98.91 | 98.91 | 98.91 | 98.91 | 0 | 10.35 | 0 | 0 | 0.01 | 1 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.25 | 0.02 | 15 | 98.52 | 0.2656 | 98.72 | 0.2027 | 0 | 0 | 0 |  | -2.297 | 98.5 | 98.49 | 98.46 | 98.45 | 0 | 18.42 | 0 | 0 | 0.01 | 2 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.25 | 0.03 | 15 | 96.64 | 0.6223 | 97.13 | 0.492 | 0 | 0 | 0 |  | -2.2 | 96.94 | 96.9 | 96.62 | 96.59 | 0 | 13.99 | 0 | 0.3917 | 0.01 | 3 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.75 | 0.01 | 15 | 98.88 | 0.05475 | 98.92 | 0.044 | 0 | 0 | 0 |  | -2.303 | 98.89 | 98.88 | 98.89 | 98.88 | 0 | 14.52 | 0 | 0 | 0.01 | 1 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.75 | 0.02 | 15 | 97.56 | 0.3954 | 98.28 | 0.7147 | 0 | 0 | 0 |  | -2.253 | 98.16 | 98.15 | 97.72 | 97.7 | 0 | 33.22 | 0 | 0.002083 | 0.01 | 2 |
| mnist | bank_init/sobel/K9 | 128 | tanh+relu@0.75 | 0.03 | 15 | 94.41 | 2.805 | 97.24 | 2.833 | 0 | 0 | 0 |  | -2.159 | 97.18 | 97.15 | 94.54 | 94.5 | 0 | 15.15 | 0 | 0.6507 | 0.01 | 3 |

## Paired comparisons

| dataset | config | hidden | lr | metric | comparison | n | mean_diff | ci_lo | ci_hi | wins | p_t | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | final_val | tanh+relu - relu | 15 | 18.34 | 11.58 | 25.84 | 15 | 0.0002695 | 0.0005389 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | test_acc_best | tanh+relu - relu | 15 | 3.193 | 2.639 | 3.769 | 15 | 4.812e-08 | 1.444e-07 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | final_val | tanh+relu - tanh | 15 | 1.113 | 0.104 | 2.344 | 10 | 0.08044 | 0.2413 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | test_acc_best | tanh+relu - tanh | 15 | 0.6813 | 0.142 | 1.295 | 11 | 0.04251 | 0.04251 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | final_val | tanh+relu@0.25 - relu | 15 | 18.83 | 12.12 | 25.9 | 15 | 0.0001995 | 0.0001995 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | test_acc_best | tanh+relu@0.25 - relu | 15 | 3.063 | 2.493 | 3.641 | 15 | 9.666e-08 | 2.9e-07 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | final_val | tanh+relu@0.75 - relu | 15 | 17.74 | 11.46 | 24.96 | 15 | 0.0002111 | 0.0004222 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.01 | test_acc_best | tanh+relu@0.75 - relu | 15 | 3.03 | 2.543 | 3.535 | 15 | 1.584e-08 | 4.751e-08 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | final_val | tanh+relu - relu | 15 | 23.66 | 17.93 | 28.34 | 15 | 6.557e-07 | 1.967e-06 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | test_acc_best | tanh+relu - relu | 15 | 2.519 | 0.708 | 4.37 | 11 | 0.02052 | 0.04104 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | final_val | tanh+relu - tanh | 15 | 2.843 | -3.305 | 7.943 | 12 | 0.3526 | 0.5031 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | test_acc_best | tanh+relu - tanh | 15 | 5.091 | 3.29 | 6.919 | 14 | 0.0001239 | 0.0003717 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | final_val | tanh+relu@0.25 - relu | 15 | 23.45 | 21.84 | 25.09 | 15 | 1.431e-13 | 4.294e-13 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | test_acc_best | tanh+relu@0.25 - relu | 15 | -0.3447 | -1.423 | 0.744 | 7 | 0.5568 | 0.5568 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | final_val | tanh+relu@0.75 - relu | 15 | 18.47 | 11.82 | 24.67 | 12 | 8.854e-05 | 0.0002656 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.02 | test_acc_best | tanh+relu@0.75 - relu | 15 | 1.975 | 0.3039 | 3.827 | 9 | 0.05294 | 0.1059 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | final_val | tanh+relu - relu | 15 | 11.9 | 5.505 | 17.83 | 12 | 0.003021 | 0.003021 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | test_acc_best | tanh+relu - relu | 15 | 3.82 | -0.2348 | 8.243 | 10 | 0.1105 | 0.1105 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | final_val | tanh+relu - tanh | 15 | 3.856 | -2.286 | 9.799 | 9 | 0.2515 | 0.5031 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | test_acc_best | tanh+relu - tanh | 15 | 6.304 | 2.403 | 10.51 | 11 | 0.01149 | 0.02298 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | final_val | tanh+relu@0.25 - relu | 15 | 12.84 | 9.209 | 16.02 | 13 | 5.281e-06 | 1.056e-05 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | test_acc_best | tanh+relu@0.25 - relu | 15 | 2.599 | -0.3654 | 5.193 | 13 | 0.09682 | 0.1936 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | final_val | tanh+relu@0.75 - relu | 15 | 7.235 | 2.724 | 11.99 | 9 | 0.01037 | 0.01037 |
| cifar10 | bank_init/laplacian/K9 | 128 | 0.03 | test_acc_best | tanh+relu@0.75 - relu | 15 | 0.102 | -4.971 | 4.762 | 10 | 0.9694 | 0.9694 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | final_val | tanh+relu - relu | 15 | 0.748 | 0.564 | 0.908 | 13 | 1.033e-06 | 1.033e-06 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | test_acc_best | tanh+relu - relu | 15 | 0.1947 | 0.1073 | 0.2773 | 12 | 0.0007249 | 0.0007249 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | final_val | tanh+relu - tanh | 15 | -0.04533 | -0.108 | 0.01733 | 6 | 0.1959 | 0.1959 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | test_acc_best | tanh+relu - tanh | 15 | -0.016 | -0.074 | 0.04533 | 5 | 0.6177 | 0.6177 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | final_val | tanh+relu@0.25 - relu | 15 | 0.8147 | 0.604 | 1 | 14 | 1.492e-06 | 1.492e-06 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | test_acc_best | tanh+relu@0.25 - relu | 15 | 0.2327 | 0.1573 | 0.3007 | 12 | 3.197e-05 | 6.395e-05 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | final_val | tanh+relu@0.75 - relu | 15 | 0.6893 | 0.4733 | 0.864 | 13 | 1.001e-05 | 1.001e-05 |
| mnist | bank_init/sobel/K9 | 128 | 0.01 | test_acc_best | tanh+relu@0.75 - relu | 15 | 0.2073 | 0.1066 | 0.3 | 13 | 0.001149 | 0.001149 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | final_val | tanh+relu - relu | 15 | 87.8 | 87.25 | 88.29 | 15 | 1.665e-28 | 3.331e-28 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | test_acc_best | tanh+relu - relu | 15 | 0.5813 | 0.4533 | 0.6947 | 15 | 3.379e-07 | 1.014e-06 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | final_val | tanh+relu - tanh | 15 | -0.404 | -0.5453 | -0.2547 | 2 | 0.0001143 | 0.0003429 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | test_acc_best | tanh+relu - tanh | 15 | -0.08533 | -0.196 | 0.01468 | 6 | 0.1429 | 0.2858 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | final_val | tanh+relu@0.25 - relu | 15 | 88.03 | 87.56 | 88.46 | 15 | 2.527e-29 | 5.053e-29 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | test_acc_best | tanh+relu@0.25 - relu | 15 | 0.67 | 0.5893 | 0.746 | 15 | 2.128e-10 | 6.384e-10 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | final_val | tanh+relu@0.75 - relu | 15 | 87.07 | 86.58 | 87.49 | 15 | 5.767e-29 | 1.73e-28 |
| mnist | bank_init/sobel/K9 | 128 | 0.02 | test_acc_best | tanh+relu@0.75 - relu | 15 | 0.3313 | 0.224 | 0.4367 | 14 | 3.66e-05 | 0.0001098 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | final_val | tanh+relu - relu | 15 | 86.98 | 86.51 | 87.35 | 15 | 1.269e-29 | 3.806e-29 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | test_acc_best | tanh+relu - relu | 15 | 1.476 | 1.111 | 1.939 | 15 | 8.842e-06 | 1.768e-05 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | final_val | tanh+relu - tanh | 15 | 1.136 | 0.708 | 1.532 | 13 | 0.0001425 | 0.0003429 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | test_acc_best | tanh+relu - tanh | 15 | 0.906 | 0.7053 | 1.109 | 15 | 6.664e-07 | 1.999e-06 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | final_val | tanh+relu@0.25 - relu | 15 | 86.45 | 86.05 | 86.81 | 15 | 2.974e-30 | 8.922e-30 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | test_acc_best | tanh+relu@0.25 - relu | 15 | 0.992 | 0.5933 | 1.501 | 14 | 0.000944 | 0.000944 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | final_val | tanh+relu@0.75 - relu | 15 | 84.21 | 82.8 | 85.47 | 15 | 2.592e-22 | 5.183e-22 |
| mnist | bank_init/sobel/K9 | 128 | 0.03 | test_acc_best | tanh+relu@0.75 - relu | 15 | 1.239 | 0.8473 | 1.686 | 15 | 6.796e-05 | 0.0001359 |
