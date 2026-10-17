from analyze_results import *
import html
NAMES={'relu':'ReLU','leaky_relu':'Leaky-ReLU','tanh':'tanh','sigmoid':'sigmoid','tanh+relu':'tanh + ReLU','tanh+leaky_relu':'tanh + leaky-ReLU','sigmoid+relu':'sigmoid + ReLU'}
ORDER=list(NAMES)
DS={'mnist':'MNIST','cifar10':'CIFAR-10'}
WIDTHS=[12,32,64,128,256,512]
def num(x,n=3):return f'{x:.{n}f}'
def signed(x,n=3):return f'{x:+.{n}f}'
def pv(x):return f'{x:.2e}' if x<.001 else f'{x:.3f}'
def ms(g,c='test_acc_best'):return f'{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}'
def ci(r):return f'[{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]'
def ref(file):return f'[{file}]({(ROOT/file).resolve().as_posix()})'
def gmain(ds,h,cfg):return subset(main,dataset=ds,hidden=h,config=cfg)
def rowpair(ds,h,cfg,b='relu'):
 return subset(pairs,dataset=ds,hidden=h,config=cfg,baseline=b,metric='test_acc_best').iloc[0]
class Doc:
 def __init__(self):self.blocks=[];self.tables=0
 def add(self,t):self.blocks.append(t.strip())
 def table(self,title,heads,rows,note=''):
  self.tables+=1;self.add(f'**Table {self.tables}. {title}**')
  self.add('| '+' | '.join(map(str,heads))+' |\n| '+' | '.join(['---']*len(heads))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows))
  if note:self.add(note)
 def save(self,name):
  text='\n\n'.join(self.blocks)+'\n';(OUT/name).write_text(text);return text
R=Doc()
R.add('# Hybrid Activations in Multilayer Perceptrons: Learning-Rate Stability, Predictive Accuracy, and Activation Allocation\n\nEnglish manuscript draft · 15 September 2026\n\nAnalysis of the version5 AWS results. All experimental numbers below were recomputed from the 32 original CSV files. This draft reports completed experiments; mechanism explanations are identified as hypotheses. The accompanying Supplementary Tables provide the complete main grid and calibration grids.')
R.add('## Abstract\n\nWe examine whether assigning different activation functions to neurons within the same hidden layer improves predictive accuracy and training stability. The experiments use two-hidden-layer multilayer perceptrons on MNIST and CIFAR-10, six hidden widths, seven activation configurations, six calibration seeds, and thirty evaluation seeds per condition. The strongest benefit is conditional resistance to late-training deterioration. On MNIST at width 128 and learning rate 0.03, all 30 ReLU runs finish below 15% validation accuracy, whereas none of the tanh–ReLU runs does; mean final validation accuracy is 10.611% and 96.761%, respectively. However, this benefit does not imply consistently higher optimally tuned accuracy. Aligning learning-rate selection with the validation-selected checkpoint reduces the width-128 MNIST accuracy difference from +0.178 to +0.029 percentage points, with a paired 95% bootstrap interval spanning zero. The frequently suggested threefold robustness advantage describes a particular upper learning-rate comparison on a coarse grid, rather than a general robustness multiplier. At fixed learning rate 0.01 and width 128, increasing the rectifier allocation from 25% to 75% improves mean MNIST checkpoint accuracy in both tanh-based hybrids. Increasing it from 50% to 75% gives only +0.014 points for tanh–ReLU, compared with +0.066 points for tanh–leaky-ReLU. CIFAR-10 and MSE experiments reveal important limits to transfer. These results support a study of activation-dependent optimization stability and allocation trade-offs, rather than an unconditional claim of hybrid superiority.')
R.add('## 1. Experimental questions and the observed hybrid benefit\n\nThe central question is whether heterogeneous activation assignment changes the learning-rate range over which useful performance can be maintained. Accuracy after model selection, accuracy at the end of training, and resistance to a large learning rate answer different questions. We therefore report them separately. ReLU is the principal homogeneous reference, while tanh, sigmoid, and leaky-ReLU provide additional controls. A hybrid must be evaluated against both of its homogeneous parents to establish what mixing contributes.')
R.table('Questions addressed by the completed tests',['Test','Available design','What it establishes'],[
['Main accuracy comparison','2 datasets × 6 widths × 7 configurations × 30 seeds','Performance at the originally calibrated LR'],
['LR calibration','8 LRs × 6 seeds per configuration and width','Validation-based hyperparameter selection'],
['Evaluation LR sweep','MNIST H=64/128; CIFAR-10 H=128; 30 seeds per LR','Sensitivity and endpoint deterioration on the tested grid'],
['Allocation ratio','MNIST H=128; 25%, 50%, 75% rectifier; LR=0.01','Ratio effect at one common LR'],
['Mask layout','Interleave, block, random at 50%','Sensitivity to activation assignment and randomization'],
['Loss/output ablation','MNIST H=64/128; CE/softmax and MSE/sigmoid','Transfer across the implemented loss/output formulations']])
R.add('### 1.1 A substantial but conditional stability advantage\n\nThe clearest result appears at MNIST width 128 and LR=0.03. ReLU achieves high validation accuracy during training and subsequently loses it. Tanh–ReLU retains high final accuracy. The behavior is also visible at width 64, although fewer ReLU seeds fail. “Calibration” in these files refers to learning-rate tuning. Predictive-probability calibration, such as reliability diagrams, expected calibration error, Brier score, or negative log-likelihood, was not measured in a form that supports such a comparison.')
rows=[]
for h in [64,128]:
 for cfg in ['relu','leaky_relu','tanh','tanh+relu','tanh+leaky_relu']:
  x=subset(lr,dataset='mnist',hidden=h,config=cfg,lr=.03)
  rows.append([h,NAMES[cfg],ms(x,'val_acc'),num(x.best_val_acc.mean()),num(x.val_drop.mean()),f'{int(x.below15.sum())}/30'])
R.table('MNIST stability at LR=0.03',['H','Activation','Final validation % ± SD','Peak validation %','Peak-to-final drop, pp','Final <15%'],rows,'Source: '+ref('lrsweep_64.csv')+' and '+ref('lrsweep_128.csv')+'. Peak is the mean of each run’s maximum validation accuracy. Drops are computed within each run before averaging. The 15% cutoff is an exploratory near-chance diagnostic for a ten-class task.')
R.add('At H=128, the mean peak-to-final decline is 85.760 points for ReLU and only 0.469 points for tanh–ReLU. Leaky-ReLU also avoids near-chance failure, but its final validation accuracy is 85.228%, substantially below tanh–ReLU. Thus avoiding catastrophic failure and preserving high accuracy are distinct properties. The hybrid with leaky-ReLU does not reproduce the tanh–ReLU high-performance result at this width.')
rows=[]
for h in [64,128]:
 for cfg in ['relu','tanh+relu']:
  x=subset(lr,dataset='mnist',hidden=h,config=cfg,lr=.03);times=[]
  for vals in x.epoch_val_acc:
   z=np.fromstring(vals,sep='|');ii=np.flatnonzero(z<15)
   if len(ii):times.append(ii[0]+1)
  cms=np.array([np.fromstring(v,sep='|').reshape(10,10) for v in x.test_confusion])
  share=cms.sum(1).max(1)/10000*100
  rows.append([h,NAMES[cfg],len(times),f'{min(times)} / {np.median(times):g} / {max(times)}' if times else 'Not observed',num(share.mean(),1)])
R.table('Timing and prediction concentration at MNIST LR=0.03',['H','Activation','Seeds ever below 15%','First such epoch: min / median / max','Mean largest predicted-class share, %'],rows,'The first crossing is observed at epoch resolution. Prediction concentration uses final test confusion matrices and is descriptive; it does not identify a causal failure mechanism.')
R.add('At width 128, ReLU first crosses below 15% between epochs 17 and 28, with a median of 21.5 among the affected runs. This temporal pattern directly supports late-training failure. Zero observed failures among 30 hybrid seeds still allows a one-sided 95% binomial upper bound of approximately 9.5% for the failure probability under independent seed sampling.')
R.add('### 1.2 What could explain the benefit?\n\nThe model assigns one activation function to each hidden neuron. A tanh–ReLU layer can therefore contain bounded tanh outputs alongside non-saturating positive ReLU outputs. This combination could alter the propagation of large activations and gradients. The experiment establishes the behavior, but does not measure the internal quantities needed to establish why it occurs.')
R.table('Mechanistic interpretation and evidence limits',['Candidate explanation','Why it is plausible','What the current results establish','Missing direct measurement'],[
['Bounded tanh outputs limit some forward signals','tanh outputs lie between −1 and 1','Tanh–ReLU preserves high MNIST accuracy at selected high LRs','Activation distributions and layerwise magnitudes'],
['Positive rectifier paths retain useful gradients','ReLU derivative is 1 for positive inputs','Increasing rectifier allocation can improve accuracy at LR=0.01','Gradient norms and feature-level contribution'],
['Negative-side gradient affects recovery','Leaky-ReLU has negative-side slope 0.1','Leaky variants avoid chance-level MNIST collapse but may lose accuracy','Inactive-unit counts and recovery trajectories'],
['Initialization and clipping interact with activation','All variants share the implemented initialization and derivative clipping','The observed effect is conditional on this implementation','Factorial initialization × clipping × activation controls'],
['Limited width constrains useful mixed features','At small widths, each activation receives few neurons','Tanh–ReLU underperforms ReLU at H=12 and H=32 on both datasets','Matched representation or feature-utilization analysis']])
R.add('Prior work motivates examining saturation, activation statistics, and initialization, but does not prove a causal explanation for these particular hybrids. See [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html), [Glorot, Bordes, and Bengio (2011)](https://proceedings.mlr.press/v15/glorot11a.html), and [He et al. (2015)](https://arxiv.org/abs/1502.01852). These are background references, not a completed novelty review.\n\nThe source clips each preactivation derivative to ±10. A weight update still has the form ΔW = −η·clip(∂L/∂z)·h, so a large preceding activation can produce a large update despite clipping. This observation motivates measuring internal dynamics; it is not evidence that gradient explosion caused the recorded failures.')
R.add('## 2. Materials, protocol, and statistical analysis')
R.table('Implemented experimental protocol',['Item','Specification'],[
['Architecture','D → H → H → 10; D=784 for MNIST and 3072 for CIFAR-10'],['Widths','H ∈ {12, 32, 64, 128, 256, 512} per hidden layer'],['Main hidden activations','Four homogeneous activations and three 50:50 hybrids'],['Training','Single-example SGD, fixed LR, 40 completed epochs; shuffled training order each epoch'],['Model selection','First epoch attaining maximum validation accuracy; selected weights evaluated after training'],['Calibration seeds','101–106; LR selected using mean final validation accuracy in the original pipeline'],['Evaluation seeds','1–30; matched seed identifiers within each comparison'],['Initialization','Weights U(−1/√fan_in, +1/√fan_in), zero biases'],['Clipping and slope','Preactivation derivatives clipped to ±10; leaky-ReLU slope 0.1'],['CE output','Linear logits followed by softmax'],['MSE output','Sigmoid outputs; implemented gradient scale differs from logged mean loss'],['Inputs','Pixel scaling to [0,1] in the adjacent dataset loaders'],['Training diagnostic','Clean accuracy on a fixed 10,000-image subset of the training split']],'Source-level details were checked in '+ref('../main.cpp')+', '+ref('../net.cpp')+', and '+ref('../sweep/select_lr.py')+'. The AWS executable was not rebuilt and matched to these sources. “Checkpoint selection” describes the computation more accurately than “early stopping,” since all 40 epochs were executed.')
R.table('Dataset provenance',['Dataset','Train','Validation','Test','Test provenance'],[['MNIST','45,000','5,000','10,000','Held out from the 60,000-image training file'],['CIFAR-10','45,000','5,000','10,000','Official CIFAR-10 test batch']],'The split seed is 12345. MNIST scores in this draft must not be described as official MNIST test-set results.')
rows=[]
for st in ['main','calib','lrsweep','layout','mse','calib_mse']:
 x=d[d.stage.eq(st)]
 rows.append([st,len(x),x.file.nunique(),', '.join(DS[v] for v in sorted(x.dataset.unique())),', '.join(str(v) for v in sorted(x.hidden.unique()))])
R.table('Run inventory after removing repeated CSV headers',['Stage','Recorded rows','Files','Datasets','Widths'],rows,'Total: 13,620 data rows in 32 CSV files. Thirty-one repeated headers were excluded. There are 690 repeated job identities across stages; they were not pooled as independent replicates. Main-grid sample size remains 30 per cell, not 60 when a sweep repeats a job.')
R.add('For all 13,620 records, both final and checkpoint confusion matrices reproduce the corresponding accuracy and macro-F1 to CSV precision. Validation histories have 40 entries and reproduce the saved maximum and selected epoch. No within-file duplicate job identities were found, and every recorded status is `ok`. These checks establish consistency of the exported records; the status field alone is not a complete numerical-health check.\n\nUnless explicitly labeled final, reported test accuracy and macro-F1 use `test_acc_best` and `test_f1_best`. Means and sample standard deviations describe seed variation. Differences use matched seeds, with 10,000 bootstrap resamples of the 30 complete pairs and percentile 95% confidence intervals. A fixed bootstrap RNG seed of 20260915 makes the current analysis reproducible; interval endpoints can differ slightly from earlier audit files. Paired t-test p-values are supplied with a Holm correction across 120 main-grid checkpoint-accuracy comparisons. A separate six-comparison correction is provided for ratio accuracy contrasts; it does not cover the entire adaptive experiment search.\n\nAccuracy differences are percentage points (pp). Relative error reduction is 100·(A_hybrid − A_baseline)/(100 − A_baseline), where A is accuracy in percent. All inferential results are conditional on one fixed split and the observed tuning procedure. The 30 training seeds are not 30 independent test datasets.')
R.add('## 3. Predictive accuracy and calibration sensitivity\n\n### 3.1 Complete width trends\n\nThe following matrices use the original learning-rate selections but consistently report validation-selected checkpoint accuracy. Complete accuracy, macro-F1, endpoint, timing, and training-gap tables for all 84 main conditions appear in the Supplementary Tables.')
for ds in DS:
 R.table(f'{DS[ds]} checkpoint test accuracy, mean ± seed SD (%)',['Activation']+[f'H={h}' for h in WIDTHS],[[NAMES[cfg]]+[ms(gmain(ds,h,cfg)) for h in WIDTHS] for cfg in ORDER],'Each cell contains 30 seeds. Main hybrids use a 50% allocation to the second activation.')
rows=[]
for ds in DS:
 for h in WIDTHS:
  p=rowpair(ds,h,'tanh+relu');rows.append([DS[ds],h,signed(p.delta),ci(p),f'{int(p.wins)}/30',pv(p.p),pv(p.p_holm)])
R.table('Paired tanh–ReLU minus ReLU accuracy at original LRs',['Dataset','H','Δ, pp','95% paired CI','Wins','Raw p','Holm p (120)'],rows,'These pointwise CIs are unadjusted; the Holm column adjusts the paired t-test family. A positive mean or nominal CI does not by itself establish a robust global advantage.')
R.add('Three patterns are consistent across the main grid. First, tanh–ReLU performs worse than ReLU at the two smallest widths on both datasets. Second, the effect does not grow monotonically with width: MNIST differences nearly vanish at widths 256 and 512, while CIFAR-10 changes sign again at width 512. Third, sigmoid–ReLU underperforms ReLU at every main-grid width on both datasets. Heterogeneity alone is therefore insufficient to explain improved performance.\n\nThe nominal highest MNIST mean is 98.289% for tanh–ReLU at H=512, compared with 98.285% for ReLU. CIFAR-10 instead has its highest mean with ReLU at H=512, 54.875%, compared with 54.711% for tanh–ReLU. These are descriptive ranks under the original calibration rule, not uniquely established winners. On CIFAR-10 H=64, the positive tanh–ReLU difference has a nominal paired interval above zero, but its Holm-adjusted p-value is approximately 0.095.')
R.add('### 3.2 The reporting endpoint changes the apparent result\n\nThe earlier `summary_extended.md` describes checkpoint selection but tabulates final-epoch test fields. This changes some conclusions, particularly on CIFAR-10. The source and CSV files provide both endpoints, so this draft can distinguish them without new training runs.')
rows=[]
for ds,h in [('mnist',32),('mnist',128),('cifar10',64),('cifar10',512)]:
 a=gmain(ds,h,'tanh+relu');b=gmain(ds,h,'relu')
 rows.append([DS[ds],h,num(b.test_acc.mean()),num(a.test_acc.mean()),signed(a.test_acc.mean()-b.test_acc.mean()),signed(a.test_acc_best.mean()-b.test_acc_best.mean())])
R.table('Final-epoch versus checkpoint interpretation',['Dataset','H','ReLU final %','Hybrid final %','Final Δ, pp','Checkpoint Δ, pp'],rows,'A validation-selected checkpoint can have lower test accuracy than the final epoch. Selection maximizes validation accuracy, not test accuracy; choosing whichever test endpoint looks better would introduce further selection bias.')
R.add('### 3.3 Aligning learning-rate calibration with checkpoint selection\n\nOriginal calibration maximizes mean final validation accuracy, even though deployment of the selected checkpoint uses the maximum validation accuracy along the trajectory. Recomputing LR selection using mean `best_val_acc` changes 17 of the 84 main selections. Existing evaluation sweeps cover four of these changed conditions; thirteen need additional evaluation runs before a fully recalibrated main grid can be reported. The MSE width-64 hybrid selection also changes and lacks matching evaluation runs.')
a=gmain('mnist',128,'tanh+relu');b0=gmain('mnist',128,'relu');b1=subset(lr,dataset='mnist',hidden=128,config='relu',lr=.01)
p0=compare(a,b0);p1=compare(a,b1)
R.table('MNIST H=128 calibration sensitivity',['Comparison','ReLU LR','Hybrid LR','ReLU checkpoint %','Hybrid checkpoint %','Δ, pp','95% CI','Relative error reduction'],[
['Original tuning',.003,.01,num(b0.test_acc_best.mean()),num(a.test_acc_best.mean()),signed(p0['delta']),ci(p0),num(p0['error_reduction'],2)+'%'],
['Checkpoint-aligned tuning',.01,.01,num(b1.test_acc_best.mean()),num(a.test_acc_best.mean()),signed(p1['delta']),ci(p1),num(p1['error_reduction'],2)+'%']])
R.add(f'After this alignment, the observed difference shrinks by {100*(1-p1["delta"]/p0["delta"]):.1f}%. The remaining +{p1["delta"]:.3f} pp corresponds to approximately {p1["delta"]*100:.1f} additional correct predictions per 10,000 images per seed on average. The paired p-value is {p1["p"]:.3f}, with {p1["wins"]}/30 positive seed differences. This retrospective comparison does not establish a reliable accuracy advantage; it also does not prove equivalence. The new rate was chosen using calibration validation records, not evaluation test accuracy, but the comparison still reuses the existing evaluation seeds.')
rows=[]
for _,c in choices[choices.changed & choices.available_evaluation_n.eq(30)].iterrows():
 oldg=gmain(c.dataset,c.hidden,c.config);newg=subset(lr,dataset=c.dataset,hidden=c.hidden,config=c.config,lr=c.lr_best)
 z=compare(newg,oldg)
 rows.append([DS[c.dataset],int(c.hidden),NAMES[c.config],f'{c.lr_final:g} → {c.lr_best:g}',num(oldg.test_acc_best.mean()),num(newg.test_acc_best.mean()),signed(z['delta']),ci(z)])
R.table('All changed main calibrations with existing evaluation coverage',['Dataset','H','Activation','LR change','Old accuracy %','Aligned accuracy %','Δ, pp','95% paired CI'],rows,'These four sensitivity checks are exploratory and are not a replacement for the missing thirteen conditions. Full changed selections are listed in the supplement.')
R.add('## 4. Quantifying learning-rate robustness\n\n### 4.1 Why a universal “threefold” statement is too broad\n\nA robustness multiplier requires a defined criterion, comparator, grid, dataset, and width. As one transparent reconstruction of the suggested 3× claim, we can compare the highest tested LR whose mean final validation accuracy remains at least 95% on the original MNIST calibration grid. At widths 64 and 128, this is 0.01 for ReLU and 0.03 for tanh–ReLU. The ratio is exactly 3 on that grid. The 95% criterion is introduced here for descriptive interpretation and was not preregistered.')
rows=[]
for h in [64,128]:
 for stage in ['calib','lrsweep']:
  x=subset(d,dataset='mnist',stage=stage,hidden=h)
  tops=[]
  for cfg in ['relu','tanh+relu']:
   gg=subset(x,config=cfg).groupby('lr').val_acc.mean();tops.append(gg[gg>=95].index.max())
  rows.append([h,stage,6 if stage=='calib' else 30,f'{tops[0]:g}',f'{tops[1]:g}',num(tops[1]/tops[0],2)+'×'])
R.table('Upper tested LR retaining mean final validation accuracy ≥95%',['H','Stage','Seeds per LR','ReLU upper LR','Tanh–ReLU upper LR','Upper-LR ratio'],rows,'The evaluation grid adds 0.005, 0.02, and 0.05. The observed ratio falls to 1.5× when the intermediate 0.02 point is available. This is a ratio of upper tested LRs, not a ratio of probabilities, accuracy, or interval widths.')
rows=[]
for ds,h in [('mnist',64),('mnist',128),('cifar10',128)]:
 for rule in ['Within 1 pp of own best','No final validation <15%']:
  vals=[]
  for cfg in ['relu','tanh+relu']:
   z=subset(windows,dataset=ds,hidden=h,config=cfg,criterion=rule).iloc[0];vals.append(z)
  a,b=vals
  rows.append([DS[ds],h,rule,f'{a.low:g}–{a.high:g} ({int(a.grid_points)})',f'{b.low:g}–{b.high:g} ({int(b.grid_points)})',num(b.high/a.high,2)+'×',f'{a.decades:.3f} / {b.decades:.3f}'])
R.table('Alternative robustness definitions on evaluation LR grids',['Dataset','H','Criterion','ReLU interval (points)','Hybrid interval (points)','Upper-LR ratio','Log10 span: ReLU / hybrid'],rows,'Intervals are the widest contiguous passing sequences on the tested grid, selected by log span. No behavior between sampled rates is assumed. Each configuration’s “own best” is its highest mean final validation score within that grid.')
R.add('For MNIST H=128, the near-optimal upper LR increases from 0.01 to 0.02, a 2× upper-bound ratio. The full log-span ratio is only 1.8239/1.5229 ≈ 1.20. For the zero-observed-failure criterion, the upper LR increases from 0.02 to 0.05, or 2.5×, and the log-span ratio is about 1.17. At H=64, the near-optimal interval is unchanged. These measures support a conditional extension of stability, but cannot be compressed into one general threefold number. Survival at a large LR can coexist with an unacceptable loss of accuracy.')
R.add('### 4.2 Dataset transfer and a non-monotonic failure pattern\n\nCIFAR-10 does not reproduce a general high-performance stability advantage. At H=128 and LR=0.01, tanh–ReLU has lower mean final validation accuracy than ReLU. At LR=0.03, it improves on a largely failed ReLU model, but remains far below the accuracy achieved at small LRs. Leaky-ReLU is a particularly important control on this dataset.')
rows=[]
for eta in [.0003,.003,.01,.03]:
 for cfg in ['relu','tanh+relu','leaky_relu']:
  z=subset(L,dataset='cifar10',hidden=128,config=cfg,lr=eta).iloc[0]
  rows.append([f'{eta:g}',NAMES[cfg],num(z.final_val),num(z.best_val),num(z['drop']),f'{int(z.fail15)}/30'])
R.table('CIFAR-10 H=128 LR sensitivity',['LR','Activation','Final validation %','Peak validation %','Drop, pp','Final <15%'],rows,'Source: '+ref('c10_lrsweep_128.csv')+'. All seven activations and eight LRs appear in the supplement.')
R.add('The CIFAR-10 tanh–ReLU condition has one final near-chance failure at LR=0.01 and nine at LR=0.03. Thus the lower failure rate than ReLU at 0.03 does not imply a uniformly expanded failure-free interval: under the contiguous zero-failure definition, its upper bound is 0.003 versus 0.01 for ReLU. In addition, the old 50% cutoff labels 561 of 1,260 calibrated CIFAR-10 main runs as “collapsed,” although none ends below 15%. The draft consequently separates task accuracy, chance-level failure, and peak-to-final deterioration.')
R.add('## 5. Comparing 25%, 50%, and 75% hybrid allocations\n\n### 5.1 Definition and controlled scope\n\nWe define the activation allocation ratio ρ as the fraction of neurons assigned to the second activation function within each hidden layer. For tanh–ReLU, ρ is the ReLU share; for tanh–leaky-ReLU, it is the leaky-ReLU share. Each neuron uses a single assigned function. The model does not compute a weighted sum of both functions at every neuron. The same allocation is applied to both hidden layers.')
R.table('Neuron counts for each tested ratio at H=128',['Second-activation share ρ','tanh share','tanh neurons per layer','Rectifier neurons per layer','Layers'],[['25%','75%',96,32,'Both hidden layers'],['50%','50%',64,64,'Both hidden layers'],['75%','25%',32,96,'Both hidden layers']],'All ratio comparisons use MNIST, H=128, interleave masks, cross-entropy, LR=0.01, and 30 paired evaluation seeds. Parameter count is 118,282 for every condition. The ratios were not independently tuned.')
rows=[]
for cfg in ['tanh+relu','tanh+leaky_relu']:
 for ratio in [.25,.5,.75]:
  x=subset(layout,config=cfg,ratio=ratio,layout='interleave')
  rows.append([NAMES[cfg],f'{ratio:.0%}',ms(x),ms(x,'test_f1_best'),num(x.best_val_acc.mean()),num(x.test_acc.mean()),num(x.best_val_epoch.mean(),1)])
R.table('Allocation ratio performance',['Hybrid','ρ','Checkpoint accuracy % ± SD','Checkpoint macro-F1 % ± SD','Peak validation %','Final test %','Selected epoch mean'],rows,'Source: '+ref('layout_128.csv')+'. The 50% interleave conditions repeat the main-grid jobs and are not independent additional evidence.')
rows=[]
for cfg in ['tanh+relu','tanh+leaky_relu']:
 for a,b in [(.5,.25),(.75,.25),(.75,.5)]:
  z=subset(ratio_pairs,config=cfg,high_ratio=a,low_ratio=b).iloc[0]
  rows.append([NAMES[cfg],f'{a:.0%} − {b:.0%}',signed(z.delta),ci(z),f'{int(z.wins)}/{int(z.ties)}/{30-int(z.wins)-int(z.ties)}',pv(z.p),pv(z.p_holm_6)])
R.table('Paired allocation contrasts',['Hybrid','Contrast','Δ accuracy, pp','95% CI','Wins/ties/losses','Raw p','Holm p (6)'],rows,'The separate Holm family includes these six ratio accuracy contrasts only. It does not adjust CIs or cover layout, LR, loss, or prior adaptive exploration.')
R.add('For tanh–ReLU, increasing the ReLU share from 25% to 50% raises mean checkpoint accuracy by 0.081 pp. Increasing it further to 75% adds only 0.014 pp, with an interval crossing zero. The latter comparison has 10 wins, 4 ties, and 16 losses across the 30 seeds, with a median difference of −0.010 pp despite its positive mean. This indicates that the small average advantage is not a consistent seed-by-seed improvement.\n\nFor tanh–leaky-ReLU, the 25% to 50% change is small and unresolved, while the 50% to 75% increase is +0.066 pp. That contrast remains below p=0.05 after the local six-comparison Holm correction, but remains an exploratory result at one fixed LR. Both 75% hybrids have the same mean checkpoint accuracy, 98.052%, to the available precision. The results suggest a plateau between 50% and 75% for tanh–ReLU and a larger benefit of the 75% allocation for tanh–leaky-ReLU within this narrow design. They do not establish a universal optimum.')
R.add('### 5.2 Comparing ratio variants with homogeneous endpoints\n\nPure tanh is available at LR=0.01, and the MNIST H=128 LR sweep supplies a pure ReLU control at that same rate. This gives a useful common-LR context for the tanh–ReLU allocation curve. Pure ReLU’s original main-grid LR=0.003 would inflate the apparent benefits of all ratio variants, so the two baselines are distinguished.')
rows=[]
for ratio in [.25,.5,.75]:
 x=subset(layout,config='tanh+relu',ratio=ratio,layout='interleave');z=compare(x,b1)
 rows.append([f'{ratio:.0%}',num(x.test_acc_best.mean()),signed(z['delta']),ci(z)])
R.table('Tanh–ReLU allocations versus pure ReLU at the common LR=0.01',['ReLU share','Hybrid accuracy %','Δ versus ReLU, pp','95% paired CI'],rows,f'Pure ReLU accuracy is {b1.test_acc_best.mean():.3f}%. Pure tanh at LR=0.01 reaches {gmain("mnist",128,"tanh").test_acc_best.mean():.3f}%. These additional endpoint comparisons are exploratory and are not included in the six-ratio correction.')
R.add('The ratio experiment measures accuracy at LR=0.01 only. It contains no high-LR sweep for 25% or 75%, so it cannot determine whether a 75% rectifier model retains the 50% model’s stability at LR=0.03. A promising next hypothesis is that a minority of tanh neurons could preserve stability while a larger rectifier allocation preserves accuracy. That hypothesis requires new measurements.')
R.add('### 5.3 Layout sensitivity at a fixed 50% allocation')
rows=[]
for cfg in ['tanh+relu','tanh+leaky_relu']:
 base=subset(layout,config=cfg,ratio=.5,layout='interleave')
 for lay in ['interleave','block','random']:
  x=subset(layout,config=cfg,ratio=.5,layout=lay);z=compare(x,base)
  rows.append([NAMES[cfg],lay,ms(x),signed(z['delta']),ci(z)])
R.table('Mask layout results at ρ=50%, LR=0.01',['Hybrid','Layout','Checkpoint accuracy % ± SD','Δ versus interleave, pp','95% CI'],rows,'Same seed identifiers are paired. Random mask generation consumes the RNG stream later used for initialization and shuffling, so nuisance randomness is not matched for those comparisons.')
R.add('No intrinsic advantage of neuron position is established. In a fully connected layer, permuting hidden neurons together with the corresponding incoming and outgoing weights preserves the represented function. Different fixed masks can still interact with a particular initialization. The present random-layout runs also change the subsequent RNG sequence, which prevents isolating mask placement alone.')
R.add('## 6. Further patterns: loss, classes, capacity, and endpoint behavior\n\n### 6.1 Loss/output transfer')
rows=[]
for h in [64,128]:
 for cfg in ['relu','tanh','tanh+relu']:
  x=subset(mse,hidden=h,config=cfg);b=subset(mse,hidden=h,config='relu');z=compare(x,b)
  rows.append([h,NAMES[cfg],f'{x.lr.iloc[0]:g}',ms(x),ms(x,'test_f1_best'),signed(z['delta']),ci(z)])
R.table('MNIST MSE/sigmoid-output checkpoint results',['H','Activation','LR','Accuracy % ± SD','Macro-F1 % ± SD','Δ versus ReLU, pp','95% CI'],rows,'These are the implemented MSE experiments, not a claim of a fully matched normalized-MSE benchmark.')
R.add('The hybrid does not outperform ReLU in the MSE results: its difference is −0.202 pp at H=64 and −0.034 pp at H=128. The H=64 hybrid would receive a different LR under checkpoint-aligned calibration, but that evaluation is unavailable. The implementation records the mean squared error over ten outputs while using a gradient corresponding to their sum, a tenfold scale difference. In addition, the loss ablation changes the output activation. Consequently, differences between CE and MSE cannot be attributed to the scalar loss function alone, and identical numerical learning rates are not equivalent update scales.')
R.add('### 6.2 Per-class patterns\n\nClass metrics were recomputed from each run’s checkpoint confusion matrix before taking seed means. This preserves per-run macro-F1 semantics. The following contrasts show where the tanh–ReLU accuracy changes occur. MNIST uses the checkpoint-aligned ReLU LR=0.01 control to avoid repeating the known calibration artifact; CIFAR-10 uses the original H=128 main LRs. These class-wise comparisons are descriptive and not multiplicity-adjusted.')
rows=[]
for ds in DS:
 a=gmain(ds,128,'tanh+relu');b=b1 if ds=='mnist' else gmain(ds,128,'relu')
 def classvals(x):
  cm=np.array([np.fromstring(v,sep='|').reshape(10,10) for v in x.test_confusion_best]);return (100*np.diagonal(cm,axis1=1,axis2=2)/cm.sum(2)).mean(0),cm[0].sum(1)
 av,support=classvals(a);bv,_=classvals(b)
 names=list(map(str,range(10))) if ds=='mnist' else ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
 for j,name in enumerate(names):rows.append([DS[ds],name,int(support[j]),num(bv[j]),num(av[j]),signed(av[j]-bv[j])])
R.table('Class recall at H=128: tanh–ReLU versus ReLU',['Dataset','Class','Test support','ReLU recall %','Hybrid recall %','Δ recall, pp'],rows,'Per-class recall concerns class-conditional accuracy; macro-F1 additionally depends on predicted-class composition. Confusion matrices cannot recover whether different models err on the same individual images.')
R.add('The small overall MNIST difference conceals opposing class effects: the aligned hybrid improves recall for digits 4 and 8 by about 0.20 pp each, while losing 0.265 pp on digit 2 and 0.137 pp on digit 3. On CIFAR-10, the same-width hybrid gains approximately 1.81 pp for dogs and 1.78 pp for frogs, while losing 1.82 pp for birds and 1.11 pp for trucks. Its small aggregate gain therefore reflects offsetting class changes, not a uniform improvement. Because these classes were examined after the overall results, these observations are hypotheses for confirmation rather than class-specific superiority claims.')
R.add('### 6.3 Capacity, generalization gap, and logged cost')
rows=[]
for ds in DS:
 for h in WIDTHS:
  x=gmain(ds,h,'relu');P=h*h+h*((784 if ds=='mnist' else 3072)+12)+10
  rows.append([DS[ds],h,f'{P:,}',num(x.test_acc_best.mean()),num(x.train_best.mean()),num(x.gap_best.mean()),num(x.seconds.mean(),1),num(x.best_val_epoch.mean(),1),f'{int(x.epoch40.sum())}/30'])
R.table('ReLU width scaling and matched-checkpoint training diagnostics',['Dataset','H','Parameters','Test %','Clean train %','Train–test gap, pp','Logged seconds','Selected epoch','Epoch 40 selected'],rows,'Parameter count is P(H,D)=H²+(D+12)H+10. Training accuracy comes from the fixed training subset at the selected checkpoint. Recorded seconds include loading, training, and evaluation under concurrent AWS workload; they are not isolated throughput or inference latency.')
R.add('Increasing width strongly improves absolute accuracy, but its marginal return decreases. The CIFAR-10 ReLU training–test gap grows from 1.609 pp at H=12 to 14.318 pp at H=512; extra capacity is fitting the training subset faster than it improves test performance. MNIST approaches a much smaller residual test error, so additional width provides relatively small absolute gains. These patterns concern the present training budget and do not identify an optimal deployment size.\n\nAcross the CIFAR-10 main grid, 239 of 1,260 runs select epoch 40. This places a nontrivial fraction of maxima at the observation boundary. It supports extending the training horizon in a follow-up, while not guaranteeing that longer training improves generalization. Conversely, an earlier selected epoch does not itself demonstrate faster convergence or saved compute because all runs execute 40 epochs.')
R.table('Selected accuracy–resource comparisons',['Dataset','Configuration','Accuracy %','Parameters','Logged seconds'],[[DS[ds],NAMES[cfg]+f', H={h}',num(gmain(ds,h,cfg).test_acc_best.mean()),f'{h*h+h*((784 if ds=="mnist" else 3072)+12)+10:,}',num(gmain(ds,h,cfg).seconds.mean(),1)] for ds,h,cfg in [('mnist',256,'relu'),('mnist',512,'tanh+relu'),('cifar10',256,'relu'),('cifar10',512,'relu')]])
R.add('On MNIST, width-256 ReLU is only 0.074 pp below the nominal width-512 hybrid mean while using 59.8% fewer parameters. On CIFAR-10, width-256 ReLU is 0.802 pp below width-512 ReLU with 53.6% fewer parameters. These are useful capacity trade-offs. Logged elapsed-time reductions of approximately 67.3% and 73.3%, respectively, are workload-specific and should not be advertised as controlled speedups.')
R.add('## 7. Discussion\n\nThe data favor a conditional interpretation of hybrid activations. Tanh–ReLU can be markedly more stable at learning rates that induce late failure in ReLU, while offering similar tuned accuracy. The two properties need not move together. Sigmoid and leaky-ReLU can also resist near-chance failure at high LRs, yet this may be accompanied by lower peak or retained performance. The meaningful comparison therefore combines failure frequency with absolute accuracy, peak-to-final decline, and calibration sensitivity.\n\nThe apparent width-128 MNIST accuracy advantage illustrates why tuning and reporting must target the same endpoint. A pipeline optimized for final validation accuracy does not necessarily provide the strongest baseline for a deployed checkpoint selected earlier in training. Correcting this mismatch removes most of the focal advantage without removing the high-LR stability observation.\n\nThe allocation experiment adds a separate contribution. Increasing the rectifier share from 25% to 75% improves mean accuracy in both tanh-based hybrids at the tested LR. The marginal effect is not the same for the two rectifiers, and a larger positive mean does not guarantee more seed-wise wins. Ratio selection and stability must consequently be studied jointly before claiming that a particular mixture is best.\n\nCross-dataset and loss results define the boundary of the claim. The CIFAR-10 sweep does not show uniformly better stability for tanh–ReLU, and the implemented MSE experiments do not replicate a hybrid advantage over ReLU. Together with the uniformly weaker sigmoid–ReLU main results, these observations argue against attributing gains simply to mixing activation functions.')
R.table('Evidence-supported patterns and appropriate claims',['Pattern','Evidence','Suitable manuscript interpretation'],[
['Late-training stability','MNIST H=128, LR=.03: 30/30 ReLU versus 0/30 hybrid failures','Tanh–ReLU prevents observed late failure in this condition'],
['Calibration dependence','Focal accuracy Δ shrinks from +.178 to +.029 pp','Tuned accuracy advantage is not established after alignment'],
['Width dependence','Small-width deficits, intermediate gains, wide-model near ties or reversals','Hybrid benefit is not monotonic in width'],
['Activation dependence','Sigmoid–ReLU trails ReLU at all 12 main dataset/width cells','Mixing alone is insufficient'],
['Allocation response','75% exceeds 25% for both tanh-based pairs at one LR','Rectifier allocation changes accuracy within the tested design'],
['Mean versus seed consistency','Tanh–ReLU 75% vs 50%: +.014 pp, 10 wins and 16 losses','A small mean gain need not be a typical-seed gain'],
['Stability versus retention','Leaky variants can stay above chance while losing >10 pp','Report both collapse and retained performance'],
['Dataset/loss limits','CIFAR-10 reversals and MSE deficits','Do not generalize the MNIST stability result to all settings'],
['Capacity trade-off','Substantial parameter increases for modest high-width gains','Report accuracy with resource measures, without a universal optimum']])
R.add('## 8. Limitations and requirements for confirmation')
R.table('Limitations and focused confirmation experiments',['Limitation','Effect on interpretation','Required confirmation'],[
['One fixed split','Seed CIs omit dataset-sampling and split uncertainty','Replicate on independent splits and official MNIST test data'],
['Endpoint-misaligned calibration','17 main LR choices change; 13 lack evaluation coverage','Evaluate missing choices with disjoint confirmation seeds'],
['Adaptive searches and multiple outcomes','Nominal effects may reflect selection','Predeclare primary contrasts and practical effect margins'],
['Only H=128/MNIST for ratio study','No evidence of an optimal ratio elsewhere','Factorial ratio × LR × width × dataset design'],
['Missing activation/gradient diagnostics','Mechanism remains hypothetical','Log norms, inactive neurons, clipping, weight scales, and class outputs'],
['Shared RNG stream for random masks','Layout cannot be isolated from subsequent randomness','Independent RNG streams and permutation-matched controls'],
['Implemented small initialization scale','Results may depend on initialization choice','Compare matched and activation-appropriate initialization'],
['MSE scale and output change','Loss effects are confounded','Verify gradients and tune equivalent update budgets'],
['40-epoch ceiling','Some best epochs remain right-censored','Extend learning curves with independently tuned schedules'],
['Concurrent elapsed timings','Cannot infer hardware-independent speed or energy savings','Measure isolated training, latency, memory, and energy'],
['No individual predictions/probabilities','No paired disagreement test or probability calibration analysis','Store labels, logits/probabilities, and per-example predictions']])
R.add('## 9. Conclusion\n\nHybrid activation assignment changes optimization behavior in a way that can be useful, particularly for retaining MNIST performance at learning rates that cause ReLU to fail late in training. The existing records support that conditional stability claim much more strongly than a general accuracy-superiority claim. A threefold figure is defensible only when explicitly attached to the coarse-grid upper-LR criterion; denser sweeps and alternative definitions produce different ratios. The 25%, 50%, and 75% allocation study shows measurable ratio dependence, with diminishing returns for tanh–ReLU between 50% and 75% and a larger 75% benefit for tanh–leaky-ReLU. Joint ratio and LR tuning, aligned checkpoint calibration, and direct internal measurements are the next steps needed to turn these observations into a broader explanatory result.')
R.add('## Data and analysis availability\n\nThe analysis leaves the original experiment files unchanged. Companion outputs contain the complete grouped metric summaries, paired main contrasts, calibration changes, LR stability summaries, ratio comparisons, and per-class summaries. Raw-file SHA-256 hashes and reconciliation results are recorded in `verification.json`. The source scripts permit regeneration of all tables. The independent recomputation reproduces all 240 accuracy and macro-F1 mean differences from the prior main-grid audit. No new training runs were performed.\n\nPrimary local documentation: '+ref('summary_extended.md')+', '+ref('rapor_ve_makale_gorusme_notlari.md')+', and '+ref('research_audit/evaluation.md')+'. The notes copied under `research_audit` are byte-identical to the root notes. The previous summary is treated as context; original CSV fields and source semantics govern the current numerical presentation.\n\nCompanion: [Supplementary Tables]('+str(OUT/'supplementary_tables.md')+').')
assert int(main[main.dataset.eq('cifar10')].below50.sum())==561
assert int(main[main.dataset.eq('cifar10')].below15.sum())==0
assert int(main[main.dataset.eq('cifar10')].epoch40.sum())==239
assert int(choices.query("stage=='calib' and changed and available_evaluation_n==30").shape[0])==4
maintext=R.save('english_manuscript_draft.md')
# Supplement: full data tables with compact repeated definitions.
S=Doc();S.add('# Supplementary Tables\n\nCompanion to “Hybrid Activations in Multilayer Perceptrons: Learning-Rate Stability, Predictive Accuracy, and Activation Allocation.”\n\nAll values are recomputed from the original CSV files. Accuracy/F1 are percentages; pp denotes percentage points. ± is sample SD across training seeds. Primary test scores use validation-selected checkpoints. Unless specified otherwise, evaluation n=30 per cell and calibration n=6. Tables in this supplement have their own numbering.')
S.add('## S1. Complete main-grid results\n\nOriginal final-validation-based LR selections are retained here; this is not the fully checkpoint-recalibrated grid. Complete means, SDs, medians, minima, and maxima for all stages are in `metric_summary.csv`.')
for ds in DS:
 for h in WIDTHS:
  rows=[]
  for cfg in ORDER:
   x=gmain(ds,h,cfg);b=gmain(ds,h,'relu')
   rows.append([NAMES[cfg],f'{x.lr.iloc[0]:g}',ms(x),ms(x,'test_f1_best'),signed(x.test_acc_best.mean()-b.test_acc_best.mean()),num(x.test_acc.mean()),num(x.gap_best.mean()),num(x.seconds.mean(),1)])
  S.table(f'{DS[ds]}, H={h}',['Activation','LR','Checkpoint accuracy ± SD','Checkpoint macro-F1 ± SD','Δ vs ReLU, pp','Final accuracy','Train–test gap, pp','Seconds'],rows)
S.add('## S2. Calibration grids\n\nEach cell shows mean final validation / mean peak validation accuracy (%), across calibration seeds 101–106. These two metrics can select different learning rates. No test scores are used for LR selection.')
for ds,st,h in cal[['dataset','stage','hidden']].drop_duplicates().sort_values(['dataset','stage','hidden']).itertuples(index=False,name=None):
 x=subset(cal,dataset=ds,stage=st,hidden=h);rates=sorted(x.lr.unique());rows=[]
 for cfg in ORDER:
  if cfg not in x.config.unique():continue
  rows.append([NAMES[cfg]]+[f'{subset(x,config=cfg,lr=eta).val_acc.mean():.3f} / {subset(x,config=cfg,lr=eta).best_val_acc.mean():.3f}' for eta in rates])
 S.table(f'{DS[ds]}, H={h}, {st}: final / peak validation',['Activation']+[f'LR={v:g}' for v in rates],rows)
S.table('All changed LR selections',['Dataset','Stage','H','Activation','Original LR','Checkpoint-aligned LR','Existing evaluation seeds'],[[DS[c.dataset],c.stage,int(c.hidden),NAMES[c.config],f'{c.lr_final:g}',f'{c.lr_best:g}',int(c.available_evaluation_n)] for _,c in choices[choices.changed].iterrows()],'Zero means no matching evaluation runs are available in the relevant completed stages. It does not mean zero accuracy.')
S.add('## S3. Complete evaluation LR grids\n\nCells show mean final validation accuracy / number of final validation scores below 15%. Each cell has 30 seeds. The score before the slash is a percentage and the count after the slash is a count out of 30. All LR results are exploratory grid descriptions.')
for ds,h in [('mnist',64),('mnist',128),('cifar10',128)]:
 x=subset(L,dataset=ds,hidden=h);rates=sorted(x.lr.unique());rows=[]
 for cfg in ORDER:
  row=[NAMES[cfg]]
  for eta in rates:
   z=subset(x,config=cfg,lr=eta).iloc[0];row.append(f'{z.final_val:.3f} / {int(z.fail15)}')
  rows.append(row)
 S.table(f'{DS[ds]}, H={h}: final validation / failure count',['Activation']+[f'{eta:g}' for eta in rates],rows)
S.add('## S4. Full parent comparisons\n\nThe following tables compare each hybrid to both of its own homogeneous parents. CIs are pointwise paired 95% bootstrap intervals. Main-grid p-values are adjusted across the complete 120-contrast accuracy family, which also includes the additional comparisons to ReLU listed in `paired_main.csv`.')
for ds in DS:
 rows=[]
 for h in WIDTHS:
  for cfg in ['tanh+relu','tanh+leaky_relu','sigmoid+relu']:
   for parent in cfg.split('+'):
    p=rowpair(ds,h,cfg,parent);rows.append([h,NAMES[cfg],NAMES[parent],signed(p.delta),ci(p),f'{int(p.wins)}/30',pv(p.p_holm)])
 S.table(f'{DS[ds]} hybrid-parent accuracy contrasts',['H','Hybrid','Parent','Δ, pp','95% CI','Wins','Holm p'],rows)
S.add('## S5. Source inventory and reproducibility\n\n`analyze_results.py` parses and verifies the original CSV files and computes statistical outputs. `build_report.py` regenerates both report documents. The bootstrap seed is 20260915 and each interval uses 10,000 paired resamples. Required analytical Python packages are numpy, pandas, and scipy. Run `python3 manuscript_draft/build_report.py` from the results directory to regenerate. The original data are read only.\n\nThe bundled runtime did not contain scipy; the available local Python environment supplied the missing statistical capability. The numerical analysis is ordinary Python and does not require a workbook engine.')
S.table('Original CSV inventory',['File','Data rows','Repeated headers ignored'],[[ref(z['file']),z['rows'],z['repeated_headers']] for z in inventory],'Exact raw-file hashes are recorded in `verification.json`. There are 12,930 distinct job identities across the 13,620 recorded rows; repeated identities across stages do not increase the independent sample count.')
supptext=S.save('supplementary_tables.md')
# Clean standalone HTML reading copies, using the same markdown source.
try:
 from markdown_it import MarkdownIt
 for name,txt in [('english_manuscript_draft',maintext),('supplementary_tables',supptext)]:
  body=MarkdownIt('commonmark').enable('table').render(txt)
  page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+name.replace('_',' ')+'</title><style>body{font-family:Georgia,serif;color:#172331;background:#f5f6f8;margin:0;line-height:1.65}main{max-width:1250px;margin:auto;background:white;padding:48px 60px}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.2}h1{font-size:36px}h2{margin-top:48px;border-top:2px solid #d9e2ed;padding-top:24px;color:#143d63}h3{margin-top:30px}p{max-width:1000px}table{display:block;overflow-x:auto;border-collapse:collapse;width:100%;font:13px/1.45 system-ui,sans-serif;margin:18px 0 28px;font-variant-numeric:tabular-nums}th{background:#183b59;color:white;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid #dfe5eb}tr:nth-child(even){background:#f2f6fa}td{vertical-align:top}a{color:#165b91}code{background:#f1f4f7;padding:2px 4px}@media(max-width:700px){main{padding:24px 18px}h1{font-size:28px}}@media print{body{background:white}main{max-width:none;padding:0}table{font-size:9px;display:table}tr{break-inside:avoid}h2,h3{break-after:avoid}}</style></head><body><main>'+body+'</main></body></html>'
  (OUT/(name+'.html')).write_text(page)
except ImportError:pass
print(json.dumps({'main_tables':R.tables,'supplementary_tables':S.tables,'main_words':len(maintext.split()),'supplementary_words':len(supptext.split()),'files':[p.name for p in OUT.iterdir() if p.suffix in ['.md','.html']]},indent=2))
