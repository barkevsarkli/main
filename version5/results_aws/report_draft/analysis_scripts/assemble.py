import re,glob,os,lib
sec={}
for f in sorted(glob.glob('out/*.md')):
    txt=open(f).read()
    for block in re.split(r'\n(?=### )',txt):
        block=block.strip('\n')
        if not block.startswith('### '): continue
        h,_,body=block.partition('\n'); sec[h[4:].strip()]=body.strip()
def filt(body,fn):
    L=body.split('\n'); return '\n'.join(L[:2]+[l for l in L[2:] if fn(l)])
# PARETO2: drop the worst-cell column, merge LR-threshold columns
P=sec['PARETO'].split('\n'); hdr=[c.strip() for c in P[0].strip('|').split('|')]
keep=[0,1,2,3]; rows=['| setting | config | calibrated test acc (ckpt) | collapse rate (<15%), all LRs | mean final val at high LRs (MNIST ≥ 0.003; CIFAR-10 ≥ 0.001) |','|---|---|---|---|---|']
for l in P[2:]:
    c=[x.strip() for x in l.strip('|').split('|')]
    hi=c[4] if c[4]!='nan' else c[6]
    rows.append('| '+' | '.join(c[:4]+[hi])+' |')
sec['PARETO2']='\n'.join(rows)
sec['CLASSMNIST']=filt(sec['CLASS'],lambda l:'| MNIST |' in l)
sec['CLASSC10']=filt(sec['CLASS'],lambda l:('| CIFAR-10 | 64 |' in l) or ('| CIFAR-10 | 256 |' in l))
for ds in ['mnist','cifar10']:
    parts=[]
    for k in sec:
        if k.startswith(f'DETAIL {ds} '):
            w=k.split()[2]; p=k.split('params=')[1]
            parts.append(f"**{('MNIST' if ds=='mnist' else 'CIFAR-10')}, {w.replace('W=','W = ')} ({p} parameters)**\n\n"+sec[k])
    sec[f'DETAILS {ds}']='\n\n'.join(parts)
sw=[]
for k in sec:
    if k.startswith('SWEEP ') : sw.append(f"**{k[6:]}**\n\n"+sec[k])
sec['SWEEPS']='\n\n'.join(sw)
cg=[]
for k in sec:
    if k.startswith('CALGRID '):
        _,ds,w=k.split(); cg.append(f"**{'MNIST' if ds=='mnist' else 'CIFAR-10'}, {w.replace('W=','W = ')}** (columns = learning rate)\n\n"+sec[k])
sec['CALGRIDS']='\n\n'.join(cg)
doc=open('tpl1.md').read()+open('tpl2.md').read()
missing=[]
def sub(m):
    key=m.group(1)
    k=key[2:] if key.startswith('T:') else key
    if k not in sec: missing.append(k); return '**MISSING '+k+'**'
    return sec[k]
doc=re.sub(r'\{\{(.+?)\}\}',sub,doc)
print('missing',missing)
os.makedirs(lib.R+'/report_draft',exist_ok=True)
open(lib.R+'/report_draft/hybrid_activation_report_draft.md','w').write(doc)
print(len(doc.split('\n')),'lines',len(doc),'chars', doc.count('\n|---'),'tables')
