import collections
import numpy as np
from _detC_load import load, SEARCH, SEARCHS

recs = load()
srch = [d for d in recs if d['LABEL'] in SEARCHS]
fold = np.array([d['FOLD'] for d in srch])

def actions(d):
    return [h['name'] for h in d['history'] if h.get('role')=='assistant_action']

def key_prev2(d):
    a=actions(d); return a[-2] if len(a)>=2 else 'NONE'
def key_prev1(d):
    a=actions(d); return a[-1] if len(a)>=1 else 'NONE'
def key_pair(d):
    a=actions(d)
    return (a[-2] if len(a)>=2 else 'NONE', a[-1] if len(a)>=1 else 'NONE')
def key_prev2_src(d):
    return (key_prev2(d), d['SRC'])

def cv_lookup(keyfn, name):
    accs=[]
    covs=[]
    for fo in [0,1,2]:
        tr=[d for d in srch if d['FOLD']!=fo]
        te=[d for d in srch if d['FOLD']==fo]
        table=collections.defaultdict(collections.Counter)
        for d in tr: table[keyfn(d)][d['LABEL']]+=1
        glob_major=collections.Counter(d['LABEL'] for d in tr).most_common(1)[0][0]
        pred={k:c.most_common(1)[0][0] for k,c in table.items()}
        correct=sum(1 for d in te if pred.get(keyfn(d), glob_major)==d['LABEL'])
        accs.append(correct/len(te))
    print(f'{name:32s} held-out acc = {np.mean(accs):.4f}')
    return np.mean(accs)

print('=== Held-out lookup-table accuracy on SEARCH subset ===')
maj=collections.Counter(d['LABEL'] for d in srch).most_common(1)[0]
print(f'{"majority baseline":32s} held-out acc = {maj[1]/len(srch):.4f}  ({maj[0]})')
cv_lookup(key_prev1, 'prev1 (last action) only')
cv_lookup(key_prev2, 'prev2 (2-steps-back) only')
cv_lookup(key_prev2_src, 'prev2 + src')
cv_lookup(key_pair, '(prev2,prev1) pair')

# pair + src
def key_pair_src(d):
    return key_pair(d)+(d['SRC'],)
cv_lookup(key_pair_src, '(prev2,prev1)+src')

# triple prev3,prev2,prev1
def key_triple(d):
    a=actions(d)
    return (a[-3] if len(a)>=3 else 'NONE', a[-2] if len(a)>=2 else 'NONE', a[-1] if len(a)>=1 else 'NONE')
cv_lookup(key_triple, '(prev3,prev2,prev1) triple')

# does prev2 relationship = the fixed cycle? print the learned prev2 map
print('\n=== learned prev2 -> label map (full data) ===')
table=collections.defaultdict(collections.Counter)
for d in srch: table[key_prev2(d)][d['LABEL']]+=1
for k,c in sorted(table.items(), key=lambda kv:-sum(kv[1].values())):
    tot=sum(c.values()); top=c.most_common(1)[0]
    print(f'  prev2={k:16s} -> {top[0]:14s} (p={top[1]/tot:.3f}, n={tot})')
