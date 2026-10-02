import collections
from _detC_load import load, SEARCH, SEARCHS

recs = load()
srch = [d for d in recs if d['LABEL'] in SEARCHS]

def actions(d):
    return [h['name'] for h in d['history'] if h.get('role')=='assistant_action']

def dist(sub, title):
    if not sub: print(f'{title}: EMPTY'); return
    c=collections.Counter(d['LABEL'] for d in sub); tot=len(sub); top=c.most_common(1)[0]
    print(f'{title}: n={tot:5d} purity={top[1]/tot:.3f}->{top[0]:14s} | '+' '.join(f'{k}={v/tot:.2f}' for k,v in c.most_common()))

# prev2 = second to last action
print('=== prev2 (2nd-to-last action) -> next SEARCH ===')
by_prev2=collections.defaultdict(list)
for d in srch:
    a=actions(d)
    p2 = a[-2] if len(a)>=2 else 'NONE'
    by_prev2[p2].append(d)
for p2,sub in sorted(by_prev2.items(), key=lambda kv:-len(kv[1])):
    if len(sub)<80: continue
    dist(sub, f'prev2={p2:14s}')

print('\n=== prev1 (last action) -> next SEARCH ===')
by_prev1=collections.defaultdict(list)
for d in srch:
    a=actions(d)
    p1 = a[-1] if len(a)>=1 else 'NONE'
    by_prev1[p1].append(d)
for p1,sub in sorted(by_prev1.items(), key=lambda kv:-len(kv[1])):
    if len(sub)<80: continue
    dist(sub, f'prev1={p1:14s}')

print('\n=== (prev2,prev1) pair -> next SEARCH (top pairs) ===')
by_pair=collections.defaultdict(list)
for d in srch:
    a=actions(d)
    p1 = a[-1] if len(a)>=1 else 'NONE'
    p2 = a[-2] if len(a)>=2 else 'NONE'
    by_pair[(p2,p1)].append(d)
for pair,sub in sorted(by_pair.items(), key=lambda kv:-len(kv[1])):
    if len(sub)<120: continue
    dist(sub, f'{str(pair):40s}')

# focus: prev2==grep_search regardless of prev1
print('\n=== SPLIT: prev2==grep_search vs not ===')
g2=[d for d in srch if (lambda a:len(a)>=2 and a[-2]=='grep_search')(actions(d))]
ng2=[d for d in srch if not (lambda a:len(a)>=2 and a[-2]=='grep_search')(actions(d))]
dist(g2,'prev2==grep_search      ')
dist(ng2,'prev2!=grep_search      ')
# and combine with prev1
print('\n  within prev2==grep, by prev1:')
sub_by=collections.defaultdict(list)
for d in g2:
    a=actions(d); sub_by[a[-1]].append(d)
for p1,sub in sorted(sub_by.items(),key=lambda kv:-len(kv[1])):
    if len(sub)<40: continue
    dist(sub, f'    prev2=grep & prev1={p1:12s}')
