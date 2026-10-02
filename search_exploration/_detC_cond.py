import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import numpy as np, pickle, collections
from _detC_load import load, SEARCH, SEARCHS

X = np.load(_p(r"work\onto\_detC_X.npy"))
meta = pickle.load(open(_p(r"work\onto\_detC_meta.pkl"),'rb'))
FN = meta['FEAT_NAMES']; y = meta['y']; fold = meta['fold']; src = meta['src']
idx = {n:i for i,n in enumerate(FN)}

def dist(mask, title):
    sub = y[mask]
    if len(sub)==0:
        print(f'{title}: EMPTY'); return
    c = collections.Counter(sub)
    tot = len(sub)
    top = c.most_common(1)[0]
    print(f'{title}: n={tot} purity={top[1]/tot:.3f}->{top[0]} | '+
          ' '.join(f'{k}={v/tot:.2f}' for k,v in c.most_common()))

print('=== prior_grep buckets (count of grep_search already in history) ===')
pg = X[:, idx['prior_grep']]
for v in [0,1,2,3]:
    dist(pg==v, f'prior_grep={v}')
dist(pg>=4, 'prior_grep>=4')

print('\n=== n_open_files buckets ===')
no = X[:, idx['n_open_files']]
for v in [0,1,2,3]:
    dist(no==v, f'n_open_files={v}')
dist(no>=4, 'n_open_files>=4')

print('\n=== prior_list buckets ===')
pl = X[:, idx['prior_list']]
for v in [0,1,2]:
    dist(pl==v, f'prior_list={v}')
dist(pl>=3,'prior_list>=3')

print('\n=== prior_read buckets ===')
pr = X[:, idx['prior_read']]
for v in [0,1,2]:
    dist(pr==v, f'prior_read={v}')
dist(pr>=3,'prior_read>=3')

print('\n=== last action name ===')
for nm,key in [('read','last_is_read'),('grep','last_is_grep'),('glob','last_is_glob'),('list','last_is_list'),('edit','last_is_edit')]:
    dist(X[:, idx[key]]==1, f'last={nm}')
# last none
none_mask = (X[:,idx['n_actions']]==0)
dist(none_mask, 'no prior actions')

print('\n=== turn_index==1 (fresh) ===')
dist(X[:,idx['turn_index']]==1, 'turn_index=1')
dist(X[:,idx['hist_len']]==0, 'hist_len=0')

print('\n=== CROSS: prior_grep==0 & n_open_files==0 ===')
dist((pg==0)&(no==0), 'pg=0 & nof=0')
dist((pg==0)&(no>=1), 'pg=0 & nof>=1')
dist((pg>=1)&(no==0), 'pg>=1 & nof=0')
dist((pg>=1)&(no>=1), 'pg>=1 & nof>=1')

print('\n=== src_au subset ===')
dist(src=='au', 'src=au (all SEARCH)')
dist(src=='sim', 'src=sim (all SEARCH)')
