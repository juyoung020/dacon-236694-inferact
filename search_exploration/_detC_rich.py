import json, re, collections
import numpy as np
from _detC_load import load, SEARCH, SEARCHS
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

recs = load()

def rescat_fine(a):
    if a is None: return 'NONE'
    rs = (a.get('result_summary') or '').lower()
    name = a['name']
    if name=='grep_search':
        if 'no match' in rs or re.search(r'\b0\s+match', rs): return 'grep_0'
        m=re.search(r'(\d+)\s+match', rs)
        if m: return 'grep_hit'
        return 'grep_other'
    if name=='glob_pattern':
        if 'no ' in rs or re.search(r'\b0\b', rs): return 'glob_0'
        return 'glob_some'
    return name

def feats(d):
    m = d['session_meta']; w = m['workspace']
    mix = w.get('language_mix') or {}
    hist = d['history']
    actions = [h for h in hist if h.get('role')=='assistant_action']
    last_a = actions[-1] if actions else None
    names = [a['name'] for a in actions]
    cnt = collections.Counter(names)
    prompt = d.get('current_prompt') or ''
    # sequence: last-2 actions
    prev1 = names[-1] if len(names)>=1 else 'NONE'
    prev2 = names[-2] if len(names)>=2 else 'NONE'
    f = {
        'turn_index': m.get('turn_index',-1),
        'elapsed': m.get('elapsed_session_sec',-1),
        'budget': m.get('budget_tokens_remaining',-1),
        'loc': w.get('loc',-1),
        'git_dirty': int(bool(w.get('git_dirty'))),
        'n_open_files': len(w.get('open_files') or []),
        'n_langs': len(mix),
        'prompt_len': len(prompt),
        'has_ext': 1 if re.search(r'\.[a-zA-Z]{1,5}\b', prompt) else 0,
        'step': d.get('STEP') if d.get('STEP') is not None else -1,
        'src_au': 1 if d['SRC']=='au' else 0,
        'hist_len': len(hist),
        'n_actions': len(actions),
        'prior_grep': cnt.get('grep_search',0),
        'prior_glob': cnt.get('glob_pattern',0),
        'prior_read': cnt.get('read_file',0),
        'prior_list': cnt.get('list_directory',0),
        'prior_edit': cnt.get('edit_file',0)+cnt.get('apply_patch',0),
        'prior_srch_total': cnt.get('grep_search',0)+cnt.get('glob_pattern',0)+cnt.get('read_file',0)+cnt.get('list_directory',0),
        'fresh': int(len(actions)==0),
    }
    # categorical -> one-hot-ish ints
    for i,nm in enumerate(['read_file','grep_search','glob_pattern','list_directory','edit_file','apply_patch','NONE']):
        f['prev1_'+nm]=int(prev1==nm)
    for i,nm in enumerate(['read_file','grep_search','glob_pattern','list_directory','NONE']):
        f['prev2_'+nm]=int(prev2==nm)
    rc = rescat_fine(last_a)
    for nm in ['grep_0','grep_hit','glob_0','glob_some','read_file','list_directory','edit_file','apply_patch','run_tests','NONE']:
        f['lastres_'+nm]=int(rc==nm)
    return f

srch = [d for d in recs if d['LABEL'] in SEARCHS]
FN = list(feats(srch[0]).keys())
X = np.array([[feats(d)[k] for k in FN] for d in srch], dtype=float)
y = np.array([d['LABEL'] for d in srch])
fold = np.array([d['FOLD'] for d in srch])

def cv(clf_factory, X, y, fold):
    accs=[]
    for fo in [0,1,2]:
        tr=fold!=fo; te=fold==fo
        clf=clf_factory()
        clf.fit(X[tr],y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return np.mean(accs)

print('n=',len(srch),'feats=',len(FN))
print('\n== RICH raw+state feature held-out SEARCH acc ==')
for depth in [3,5,7,10]:
    a=cv(lambda: DecisionTreeClassifier(max_depth=depth,min_samples_leaf=40,random_state=0),X,y,fold)
    print(f'DTree depth={depth}: {a:.4f}')
a=cv(lambda: RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=20,random_state=0,n_jobs=-1),X,y,fold)
print(f'RandomForest: {a:.4f}')
a=cv(lambda: GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=0),X,y,fold)
print(f'GradBoost: {a:.4f}')

# importances
rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=20,random_state=0,n_jobs=-1).fit(X,y)
imp=sorted(zip(FN,rf.feature_importances_),key=lambda x:-x[1])
print('\n== RF importances top15 ==')
for n,v in imp[:15]: print(f'  {n}: {v:.4f}')

# print a shallow tree (depth 4) trained on fold!=0
tr=fold!=0
clf=DecisionTreeClassifier(max_depth=4,min_samples_leaf=100,random_state=0).fit(X[tr],y[tr])
print('\n== Depth-4 tree (trained on folds 1,2) ==')
print(export_text(clf, feature_names=FN, max_depth=4))
