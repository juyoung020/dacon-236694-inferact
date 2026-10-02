import collections, re
import numpy as np
from _detC_load import load, SEARCH, SEARCHS
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

recs = load()
srch = [d for d in recs if d['LABEL'] in SEARCHS]

ACTS=['read_file','grep_search','glob_pattern','list_directory','edit_file','apply_patch',
      'write_file','run_tests','run_bash','lint_or_typecheck','plan_task','ask_user','web_search','respond_only','NONE']
def actions(d):
    return [h['name'] for h in d['history'] if h.get('role')=='assistant_action']

def feats(d):
    m=d['session_meta']; w=m['workspace']; mix=w.get('language_mix') or {}
    a=actions(d); cnt=collections.Counter(a)
    prompt=d.get('current_prompt') or ''
    p1=a[-1] if len(a)>=1 else 'NONE'
    p2=a[-2] if len(a)>=2 else 'NONE'
    p3=a[-3] if len(a)>=3 else 'NONE'
    f={
      'turn_index':m.get('turn_index',-1),'elapsed':m.get('elapsed_session_sec',-1),
      'budget':m.get('budget_tokens_remaining',-1),'loc':w.get('loc',-1),
      'git_dirty':int(bool(w.get('git_dirty'))),'n_open_files':len(w.get('open_files') or []),
      'n_langs':len(mix),'prompt_len':len(prompt),'has_ext':1 if re.search(r'\.[a-zA-Z]{1,5}\b',prompt) else 0,
      'src_au':1 if d['SRC']=='au' else 0,'n_actions':len(a),
      'prior_grep':cnt.get('grep_search',0),'prior_read':cnt.get('read_file',0),
      'prior_glob':cnt.get('glob_pattern',0),'prior_list':cnt.get('list_directory',0),
      'p1':ACTS.index(p1) if p1 in ACTS else -1,
      'p2':ACTS.index(p2) if p2 in ACTS else -1,
      'p3':ACTS.index(p3) if p3 in ACTS else -1,
    }
    return f

FN=list(feats(srch[0]).keys())
X=np.array([[feats(d)[k] for k in FN] for d in srch],dtype=float)
y=np.array([d['LABEL'] for d in srch])
fold=np.array([d['FOLD'] for d in srch])
src=np.array([d['SRC'] for d in srch])

def cv(fac):
    accs=[]
    for fo in [0,1,2]:
        tr=fold!=fo; te=fold==fo
        c=fac(); c.fit(X[tr],y[tr]); accs.append(accuracy_score(y[te],c.predict(X[te])))
    return np.mean(accs)

print('=== DEFINITIVE raw-feature held-out SEARCH accuracy ===')
print('features:',FN)
print(f'DTree d6        : {cv(lambda:DecisionTreeClassifier(max_depth=6,min_samples_leaf=40,random_state=0)):.4f}')
print(f'HistGBM         : {cv(lambda:HistGradientBoostingClassifier(max_iter=400,max_depth=6,learning_rate=0.08,random_state=0)):.4f}')

# per-fold sim-only accuracy (the confusable subset)
print('\n=== restricting to SRC=sim (the confusable core) ===')
sim_mask=src=='sim'
accs=[]
for fo in [0,1,2]:
    tr=(fold!=fo); te=(fold==fo)&sim_mask
    c=HistGradientBoostingClassifier(max_iter=400,max_depth=6,learning_rate=0.08,random_state=0)
    c.fit(X[tr],y[tr]); accs.append(accuracy_score(y[te],c.predict(X[te])))
print(f'HistGBM sim-only held-out: {np.mean(accs):.4f}')

# ===== au vs sim: does prev2 rule hold in both? =====
print('\n=== prev2 rule by source ===')
for s in ('au','sim'):
    sub=[d for d in srch if d['SRC']==s]
    tab=collections.defaultdict(collections.Counter)
    for d in sub:
        aa=actions(d); k=aa[-2] if len(aa)>=2 else 'NONE'; tab[k][d['LABEL']]+=1
    print(f' SRC={s}:')
    for k in ['read_file','grep_search','glob_pattern','list_directory','NONE']:
        c=tab.get(k);
        if not c: continue
        tot=sum(c.values()); top=c.most_common(1)[0]
        print(f'   prev2={k:14s} -> {top[0]:14s} p={top[1]/tot:.3f} n={tot}')

# ===== generation drift: label dist vs SEQ position within sim =====
print('\n=== generation drift: sim records sorted by (DAY,SEQ) into 5 buckets ===')
sim=[d for d in srch if d['SRC']=='sim']
sim.sort(key=lambda d:(d['SEQ'] or '', d['SUB'] or 0))
nb=5; B=len(sim)//nb
for b in range(nb):
    chunk=sim[b*B:(b+1)*B] if b<nb-1 else sim[b*B:]
    c=collections.Counter(d['LABEL'] for d in chunk); tot=len(chunk)
    print(f'  bucket{b}: '+' '.join(f'{k}={c[k]/tot:.2f}' for k in SEARCH))

# fold vs label (should be balanced if folds random)
print('\n=== label dist by FOLD (drift check) ===')
for fo in [0,1,2]:
    c=collections.Counter(y[fold==fo]); tot=int((fold==fo).sum())
    print(f'  fold{fo}: '+' '.join(f'{k}={c[k]/tot:.3f}' for k in SEARCH))
