from pathlib import Path
import os, json
os.environ['MPLCONFIGDIR']='/private/tmp/activation-audit-mpl'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'all_metrics.csv'); p=pd.read_csv(R/'paired_comparisons.csv')
main=s[s.stage.eq('main')]
order=['relu','leaky_relu','tanh','sigmoid','tanh+relu','tanh+leaky_relu','sigmoid+relu']
colors=['#263b53','#538fa1','#9786ae','#b09e6a','#007e75','#7fa940','#cb705a']
fig,axes=plt.subplots(1,3,figsize=(16,4.9),gridspec_kw={'width_ratios':[1,1,1.2]})
for ax,ds,title in zip(axes[:2],['mnist','cifar10'],['MNIST: held-out training-file subset','CIFAR-10: official test set']):
    for c,col in zip(order,colors):
        g=main[(main.dataset==ds)&(main.config==c)].sort_values('hidden')
        ax.plot(g.hidden,g.test_acc_best_mean,'o-',ms=4,lw=1.5,color=col,label=c)
    ax.set_xscale('log',base=2); ax.set_xticks([12,32,64,128,256,512],['12','32','64','128','256','512'])
    ax.set_xlabel('Hidden width (each of two layers)'); ax.set_ylabel('Mean checkpoint test accuracy (%)'); ax.set_title(title,fontsize=11)
    ax.grid(alpha=.16)
ax=axes[2]
vals=[.1776666667,.0293333333]; lo=[.1413333333,-.0176666667]; hi=[.213675,.082]
ax.errorbar(vals,[1,0],xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='o',capsize=5,color='#007e75',ms=7)
ax.axvline(0,color='#7a8490',ls='--',lw=1)
ax.set_yticks([1,0],['Original LR selection\nReLU LR = 0.003','Aligned checkpoint selection\nReLU LR = 0.01'])
ax.set_ylim(-.7,1.7); ax.set_xlim(-.045,.25)
ax.set_xlabel('tanh+ReLU − ReLU accuracy (percentage points)')
ax.set_title('MNIST width 128: baseline sensitivity',fontsize=11)
ax.text(.1777,1.22,'+0.178 pp',ha='center',fontsize=10)
ax.text(.0293,.22,'+0.029 pp',ha='center',fontsize=10)
ax.grid(axis='x',alpha=.16)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=7,frameon=False,fontsize=9,bbox_to_anchor=(.5,.025))
fig.suptitle('Validation-selected checkpoints change the interpretation of hybrid activations',fontsize=14,y=.98)
fig.text(.5,.005,'30 paired seeds per configuration. Error bars: pointwise 95% paired bootstrap intervals. Curves retain the original calibrated learning rates.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.1,1,.93],w_pad=2)
fig.savefig(R/'audit_overview.png',dpi=180); fig.savefig(R/'audit_overview.svg')

intro='''Experimental evaluation of hybrid-activation MLPs

The supplied prompt contains placeholders. This evaluation therefore uses the 32 result CSVs in the shared results_aws workspace and the adjacent experimental source. The inferred objective is to assess whether heterogeneous hidden activations improve accuracy, macro-F1, optimization stability, or computational trade-offs relative to homogeneous parents. ReLU is the primary baseline; each hybrid is also compared with both parents. No raw result or existing report was modified.

The strongest defensible finding is a learning-rate-dependent stability benefit on MNIST, with limited evidence of a generally superior hybrid classifier. The nominal highest main-grid checkpoint accuracy is tanh+ReLU at width 512 on MNIST (98.289% ± 0.087% seed SD), effectively tied with ReLU (98.285% ± 0.064%). CIFAR-10 instead favors ReLU at width 512 descriptively (54.875% ± 0.469%; macro-F1 54.577%). Its difference from tanh+ReLU is not resolved by the paired interval. Rankings are descriptive and conditional on the original calibration rule, one data split, and a 40-epoch budget.

A critical endpoint mismatch changes the central MNIST width-128 result. The existing summary calls its scores early-stopping results but reads final-epoch test_acc/test_f1. The runner also stores test_acc_best/test_f1_best from the maximum-validation-accuracy checkpoint. Moreover, calibration selects learning rates by final val_acc. For ReLU width 128, calibrating on best_val_acc selects 0.01 rather than 0.003. Comparing checkpoint scores at those aligned rates reduces the tanh+ReLU advantage from +0.178 percentage points (95% paired bootstrap CI +0.141 to +0.214) to +0.029 (−0.018 to +0.082; paired t p=0.266; 17/30 wins). Roughly 83.5% of the apparent difference disappears. The remaining difference is about 2.9 additional correct classifications per 10,000 images per seed, on average. Failure to reject zero does not establish equivalence.

For resource-constrained selection, width-256 ReLU is a reasonable provisional compromise on both datasets. On MNIST it reaches 98.214%, only 0.074 percentage points below the nominal width-512 hybrid winner, using 269,322 versus 669,706 parameters (59.8% fewer) and 482.5 versus 1,474.4 logged seconds (67.3% less). On CIFAR-10 width-256 ReLU reaches 54.074%, 0.802 points below width-512 ReLU, with 855,050 versus 1,841,162 parameters (53.6% fewer) and 2,240.3 versus 8,400.7 seconds (73.3% less). A unique efficiency optimum requires a deployment objective and direct latency or energy measurements.

Protocol and integrity

The network is D→H→H→10, D=784 or 3072, with H∈{12,32,64,128,256,512}. Training uses single-example SGD, 40 completed epochs, zero initial biases, a shared fixed split, shuffled epochs, and elementwise clipping of preactivation derivatives at ±10. Leaky-ReLU slope is 0.1. CE uses linear logits and softmax; the MSE runs use sigmoid outputs. Main hybrids assign 50% of each hidden layer to each activation. The implementation selects a checkpoint after training all epochs; it does not stop computation early.

The audit found 13,620 data rows, all status=ok, after removing 31 repeated CSV header lines. There were no duplicated job keys within individual files and no missing checkpoint scores. The 2,520 main runs comprise 2 datasets × 6 widths × 7 configurations × 30 seeds. Calibration uses seeds 101–106; evaluation uses 1–30. There are 690 repeated job identities across stages, so pooling stages as independent replication would inflate n. Confusion matrices for every row contain 10,000 predictions and reproduce checkpoint accuracy and macro-F1 within CSV rounding. All validation histories have 40 entries and reproduce the stored maximum and selected epoch. Class-support vectors are consistent within each dataset; this verifies composition, not image identity or absence of duplicate images.

MNIST is split 45,000/5,000/10,000 from the 60,000-image training file; its reported test accuracy must not be described as accuracy on the official MNIST test set. CIFAR-10 uses 45,000 training and 5,000 validation examples from its training set and the official 10,000-image test batch. Source-level observations below refer to the adjacent source; the AWS executable was not rebuilt and matched to that source during this audit.

Quantitative interpretation

Unless explicitly marked final, accuracy and F1 below use the validation-selected checkpoint. SD describes variation across seeds, not uncertainty across new dataset samples. For paired differences d_s=A_variant,s−A_baseline,s, the estimated effect is mean(d_s); confidence limits are percentiles of 10,000 resamples of complete seed pairs. Relative accuracy change is 100×(A_variant−A_baseline)/A_baseline; relative error reduction is 100×(A_variant−A_baseline)/(100−A_baseline), with accuracies expressed in percent. These are different quantities. The original width-128 MNIST gain represents 0.182% relative accuracy improvement or 8.30% error reduction; after aligned calibration those become approximately 0.030% and 1.47%.

At widths 12 and 32 on MNIST, tanh+ReLU underperforms ReLU by 0.307 and 0.186 points at selected checkpoints. At 64, 256, and 512 the differences are +0.004, −0.001, and +0.003 points. Thus the results do not support a monotonic width-dependent emergence of hybrid superiority. On CIFAR-10 its differences versus ReLU are −0.550, −0.385, +0.363, +0.178, +0.153, and −0.164 points in ascending width order. The width-64 effect is nominally positive (95% CI +0.155 to +0.577), but its Holm-adjusted p is 0.095 across the 120 audited main-grid checkpoint-accuracy contrasts. At widths 128 and 256 its pointwise intervals include zero. Beating tanh alone is insufficient evidence that mixing is better than the strongest homogeneous parent.

Sigmoid+ReLU underperforms ReLU at every main-grid width on both datasets. Its MNIST deficit reaches 0.870 points at width 64 (approximately 35.9% relative error increase). Nevertheless, it can improve on homogeneous sigmoid, especially on CIFAR-10. Macro-F1 broadly follows accuracy, but the two criteria are not interchangeable: CIFAR-10 width-64 tanh+leaky-ReLU has slightly higher accuracy than leaky-ReLU (51.288 vs 51.270) and lower macro-F1 (50.847 vs 50.904). The companion class table provides seed means and SDs for precision, recall, and F1 for every main, MSE, and layout condition. A confusion matrix does not provide predictive-probability calibration, AUROC, or example-level cross-model error agreement.

Anomalies and mechanisms

At MNIST width 128 and LR=0.03, all 30 ReLU runs finish below 15% validation accuracy, with mean 10.611%, despite mean best validation accuracy of 96.371%. Tanh+ReLU at the same width and LR finishes at 96.761%, with no runs below 15%. Width-64 ReLU has 12/30 such failures at LR=0.03, versus 0/30 for tanh+ReLU. This is a reproducible late-training instability and a strong conditional robustness result. It does not establish superior optimal accuracy, and an all-zero observed failure count still permits an approximately 9.5% one-sided 95% binomial upper bound under independent-seed sampling.

ReLU’s zero negative-side derivative can prevent recovery of inactive units, whereas leaky-ReLU retains a negative-side gradient and tanh bounds forward activations. These mechanisms plausibly contribute to stability differences. The logs contain no activation histograms, gradient norms, clipping frequencies, or dead-unit counts, so neither dead ReLUs nor gradient explosion is established as the cause. Clipping dL/dz does not bound weight updates when the preceding activation is large: ΔW=−η·clip(dL/dz)·h. The source’s non-finite check tests only the first output for NaN; status=ok is not a comprehensive numerical-health certificate.

The report’s final-validation threshold of 50% labels 561/1,260 calibrated CIFAR-10 main runs as collapsed, although none finishes below 15%. This confuses limited accuracy with optimization failure. The 15% level used here is only a transparent near-chance diagnostic, not a universally validated threshold. Report chance-level failure, substantial deterioration from the run’s peak, and task-relative performance degradation separately. High-LR survival can also be misleading: MNIST width-128 leaky-ReLU at LR=0.03 finishes around 85.228% validation accuracy, while its best validation mean is 96.365%. Remaining above 50% does not mean retaining useful performance.

The implemented initializer samples U(−1/√fan_in,+1/√fan_in), giving Var(W)=1/(3 fan_in). Standard He variance for ReLU is 2/fan_in, six times larger. This is the Linear-default-style scale, not ReLU-gain He initialization. Identical weight distributions across activations control one implementation variable but do not establish equally well-tuned baselines. Sigmoid derivatives are at most 1/4, tanh derivatives approach zero in saturation, and input scaling to [0,1] is not centering. Their interaction with the small initialization scale and finite budget is a plausible explanation for optimization differences. These hypotheses are consistent with the activation and initialization analyses of [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) and [He et al. (2015)](https://arxiv.org/abs/1502.01852); those papers do not validate the present hybrid-specific causal claims or constitute matched performance benchmarks.

Layout and loss ablations

At width 128, increasing the rectifier fraction from 0.5 to 0.75 raises tanh+ReLU checkpoint accuracy by 0.014 points (95% CI −0.017 to +0.047), and tanh+leaky-ReLU by 0.066 points (+0.025 to +0.109). Both 0.75 variants average 98.052%; this is a descriptive tie. Ratios were not independently calibrated, and the latter interval is unadjusted for layout searches. Interleave, block, and random arrangements do not establish an intrinsic positional advantage. In a dense MLP, neuron permutations with corresponding weight permutations preserve the represented function. Random masks additionally consume the same RNG stream used later for weights and shuffles, so same seed does not match these nuisance variables across random-layout comparisons.

MSE does not replicate a hybrid advantage over ReLU. At width 64, tanh+ReLU scores 97.435% versus ReLU 97.637% (difference −0.202 points; 95% CI −0.267 to −0.138). At width 128 it scores 97.964% versus 97.998% (−0.034; −0.077 to +0.009). The width-64 checkpoint-based calibration would select a different hybrid LR, so that deficit also requires endpoint-aligned confirmation. The existing extended summary omits the available width-64 MSE results. MSE comparisons change both objective and output activation. Furthermore, the source logs (1/10)Σ_j error_j² but backpropagates 2·error_j rather than (2/10)·error_j. This is a tenfold gradient-scale mismatch to the logged objective, which must be resolved before interpreting equal numerical learning rates across losses.

Efficiency, convergence, and validity limits

The parameter count including biases is P(H,D)=H²+(D+12)H+10. At fixed width all activation variants have equal trainable parameter counts, but nonlinear-function cost can differ. Logged seconds include data loading, training, per-epoch validation and clean-training evaluation, and final evaluations. Logs show up to 190 workers on 192 cores. These are run elapsed times under that workload, not inference latency or isolated training throughput. CIFAR-10 width-512 sigmoid+ReLU appears 38.8% faster than ReLU, but this cannot be attributed to activation mechanics without controlled timing. Repeated width-128 jobs across main and LR stages reproduce accuracy exactly and timing within a few percent; that supports local repeatability, not cross-width hardware comparability.

Checkpoint training accuracy is reconstructed from the clean-training history at the selected epoch, using a fixed 10,000-image training subset. It must not be combined with final-epoch training accuracy or online training accuracy to infer overfitting. ReLU’s CIFAR-10 checkpoint train–test gap grows from 1.609 points at width 12 to 14.318 at width 512; MNIST width-512 ReLU’s gap is 1.714 points. This supports investigating generalization limits on CIFAR-10, although the training subset is not an independent validation sample. Best epochs average roughly 32–38 on CIFAR-10, and 239/1,260 main runs select epoch 40. The right-censored budget warrants longer learning curves. Selecting an earlier maximum alone does not establish faster convergence or saved computation.

Thirty training seeds quantify stochastic training variation conditional on a single split. They are not 30 independent test sets or 300,000 independent test examples. The bootstrap intervals do not include split uncertainty, calibration-seed uncertainty, or adaptive configuration selection. With 10,000 test images, an individual-model binomial SE is roughly 0.13 percentage points near 98.3% accuracy and 0.50 points near 54.9%; paired model comparisons require per-example joint predictions to exploit shared errors. Confusion matrices cannot recover those disagreements. Multiple widths, activations, LRs, layouts, endpoints, and losses create substantial multiplicity; the Holm correction supplied covers 120 main checkpoint-accuracy comparisons only. F1, layout, MSE, and LR analyses remain exploratory. Disjoint calibration seeds are a strength, but six calibration seeds can yield unstable LR choices, and repeated use of the same validation examples does not remove selection optimism. Seventeen of 84 main calibration choices change when switching from final to maximum validation accuracy (18 of 90 including MSE). Unavailable evaluation runs at newly selected rates require further experiments.

Prioritized next experiments

1. Align calibration and reporting with the deployed endpoint. Select LR by mean best_val_acc if the best checkpoint is deployed; report test_acc_best/test_f1_best, selected epoch, and checkpoint-matched training metrics. Keep final-epoch scores as a separately named endpoint. Re-evaluate changed selections on untouched seeds; available width-64/128 sweeps permit retrospective checks but not independent confirmation.
2. Validate implementation semantics. Add finite-difference checks of CE and normalized-MSE gradients, scan every activation/weight/gradient for non-finite values, document derivative clipping, and compare predictions and updates against a reference implementation. Record executable, compiler, source, dataset, split, and initial-weight hashes. Decouple RNG streams for initialization, masks, and example order.
3. Test the strongest robustness claim directly. At widths 64 and 128, densely sample LRs around 0.01–0.03 using independent seeds and identical initial weights/order. Measure dead-unit fraction, activation distributions, gradient norms, clipping frequency, prediction-class entropy, peak-to-final deterioration, and time of failure. Include ReLU, leaky-ReLU, and tanh+ReLU.
4. Run a factorial initialization × activation × LR study. Include the present scale, activation-appropriate He/Glorot initialization, and centered inputs. Give all variants equal tuning budgets. Separately vary clipping and output/loss formulation. Only then attribute improvements to activation heterogeneity.
5. Replicate across data splits and the official MNIST test set. Predeclare the main paired comparison, multiplicity family, and a practical margin such as 0.1 percentage points on MNIST if relevant to the application. Store per-example labels, predictions, and probabilities for paired disagreement tests, hierarchical uncertainty, class-level diagnostics, NLL, Brier score, and calibration analysis.
6. Resolve capacity and cost trade-offs. Extend CIFAR-10 training beyond 40 epochs with independently tuned schedules and weight decay; test a CNN reference if the goal is image-classification performance. Measure isolated training time, time to a fixed validation target, inference latency distributions, memory, and energy at widths 128/256/512. Retune ratios 0.25/0.5/0.75 and use permutation-matched controls for layout.

Complete main-grid comparison

Each table row below is one configuration at its original calibrated LR and n=30. Acc/F1 are checkpoint percentages with sample SD. Δ is checkpoint accuracy minus ReLU at the same dataset and width; rel. error is percent error reduction versus that baseline. Time is the logged total run mean. Full final/selected validation, train, test, epoch, timing, and loss summaries for all stages are in all_metrics.csv, including means, SDs, medians, minima and maxima. Paired intervals, wins, raw p-values and the primary Holm corrections are in paired_comparisons.csv. LR cell counts and final/best metrics are in lr_diagnostics.csv; derived per-class precision/recall/F1 are in class_metrics.csv. These companion tables are analytical outputs, not additional independent observations.
'''
lines=[intro]
for ds in ['mnist','cifar10']:
    for w in [12,32,64,128,256,512]:
        g=main[(main.dataset==ds)&(main.hidden==w)].set_index('config').loc[order]
        b=g.loc['relu','test_acc_best_mean']
        lines += [f'\n{ds.upper()}, width {w}; {int(g.parameters.iloc[0]):,} parameters\n', '| Configuration | LR | Acc ± SD (%) | F1 ± SD (%) | Δ acc (pp) | Rel. error reduction (%) | Final acc (%) | Selected epoch mean | Seconds mean |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c,a in g.iterrows():
            delta=a.test_acc_best_mean-b
            lines.append(f'| {c} | {a.lr:g} | {a.test_acc_best_mean:.3f} ± {a.test_acc_best_std:.3f} | {a.test_f1_best_mean:.3f} ± {a.test_f1_best_std:.3f} | {delta:+.3f} | {100*delta/(100-b):+.2f} | {a.test_acc_mean:.3f} | {a.best_val_epoch_mean:.1f} | {a.seconds_mean:.1f} |')
lines += ['\nLayout and MSE checkpoint comparisons\n','| Stage | Width | Config | Layout | Rectifier fraction | LR | n | Acc ± SD (%) | F1 (%) |','|---|---:|---|---|---:|---:|---:|---:|---:|']
for _,a in s[s.stage.isin(['layout','mse'])].iterrows():
    lines.append(f'| {a.stage} | {a.hidden} | {a.config} | {a.layout} | {a.ratio:g} | {a.lr:g} | {a.n} | {a.test_acc_best_mean:.3f} ± {a.test_acc_best_std:.3f} | {a.test_f1_best_mean:.3f} |')
(R/'evaluation.md').write_text('\n'.join(lines))
print('Wrote evaluation.md and overview PNG/SVG')
