import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, re, collections
import numpy as np
from _detC_load import load, SEARCH, SEARCHS
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

recs = load()

LANGS = ['py','ts','tsx','js','jsx','go','rs','java','rb','css','html','json','yaml','yml','md','dockerfile','toml','sh','c','cpp','kt','swift','php']
TIERS = ['free','pro','enterprise','team']
RESULTS = None

EXT_RE = re.compile(r'\.[a-zA-Z]{1,5}\b')

def dom_lang(mix):
    if not mix: return 'none'
    return max(mix.items(), key=lambda kv: kv[1])[0]

def result_cat(rs):
    if rs is None: return 'none'
    s = rs.lower()
    if s.startswith('ok') or 'pass' in s or 'passed' in s: return 'ok'
    if 'fail' in s or 'error' in s or 'err' in s: return 'fail'
    if 'match' in s: return 'match'
    if 'no ' in s or 'not found' in s or '0 ' in s: return 'empty'
    return 'other'

def feats(d):
    m = d['session_meta']
    w = m['workspace']
    mix = w.get('language_mix') or {}
    hist = d['history']
    actions = [h for h in hist if h.get('role') == 'assistant_action']
    users = [h for h in hist if h.get('role') == 'user']
    last_a = actions[-1] if actions else None
    last_name = last_a['name'] if last_a else 'NONE'
    last_res = result_cat(last_a.get('result_summary') if last_a else None)
    prompt = d.get('current_prompt') or ''
    # action history counts
    names = [a['name'] for a in actions]
    cnt = collections.Counter(names)
    dl = dom_lang(mix)
    f = {
        'turn_index': m.get('turn_index', -1),
        'elapsed': m.get('elapsed_session_sec', -1),
        'budget': m.get('budget_tokens_remaining', -1),
        'loc': w.get('loc', -1),
        'git_dirty': int(bool(w.get('git_dirty'))),
        'n_open_files': len(w.get('open_files') or []),
        'n_langs': len(mix),
        'dom_lang_frac': max(mix.values()) if mix else 0.0,
        'user_tier': TIERS.index(m['user_tier']) if m.get('user_tier') in TIERS else -1,
        'lang_pref_ko': 1 if m.get('language_pref') == 'ko' else 0,
        'ci_passed': 1 if w.get('last_ci_status') == 'passed' else 0,
        'ci_failed': 1 if w.get('last_ci_status') == 'failed' else 0,
        'ci_none': 1 if w.get('last_ci_status') in (None,'none') else 0,
        'prompt_len': len(prompt),
        'prompt_words': len(prompt.split()),
        'has_ext': 1 if EXT_RE.search(prompt) else 0,
        'has_slash': 1 if '/' in prompt else 0,
        'has_quote': 1 if ('"' in prompt or "'" in prompt or '`' in prompt) else 0,
        'step': d.get('STEP', -1) if d.get('STEP') is not None else -1,
        'sub': d.get('SUB', -1) if d.get('SUB') is not None else -1,
        'src_au': 1 if d['SRC'] == 'au' else 0,
        'hist_len': len(hist),
        'n_actions': len(actions),
        'n_users': len(users),
        'last_is_read': int(last_name == 'read_file'),
        'last_is_grep': int(last_name == 'grep_search'),
        'last_is_glob': int(last_name == 'glob_pattern'),
        'last_is_list': int(last_name == 'list_directory'),
        'last_is_edit': int(last_name in ('edit_file','apply_patch')),
        'last_res_ok': int(last_res == 'ok'),
        'last_res_fail': int(last_res == 'fail'),
        'last_res_empty': int(last_res == 'empty'),
        'prior_grep': cnt.get('grep_search', 0),
        'prior_glob': cnt.get('glob_pattern', 0),
        'prior_read': cnt.get('read_file', 0),
        'prior_list': cnt.get('list_directory', 0),
        'dom_lang': LANGS.index(dl) if dl in LANGS else -1,
    }
    return f

# build feature matrix for SEARCH subset only
srch = [d for d in recs if d['LABEL'] in SEARCHS]
FEAT_NAMES = list(feats(srch[0]).keys())
X = np.array([[feats(d)[k] for k in FEAT_NAMES] for d in srch], dtype=float)
y = np.array([d['LABEL'] for d in srch])
fold = np.array([d['FOLD'] for d in srch])
src = np.array([d['SRC'] for d in srch])

print('SEARCH n=', len(srch), 'features=', len(FEAT_NAMES))
maj = collections.Counter(y).most_common(1)[0]
print('majority baseline (all):', round(maj[1]/len(y), 4), maj[0])

# Held-out CV using folds
def cv_tree(X, y, fold, depth, feat_subset=None):
    accs = []
    preds = np.empty(len(y), dtype=object)
    for fo in [0,1,2]:
        tr = fold != fo
        te = fold == fo
        Xtr, Xte = X[tr], X[te]
        if feat_subset is not None:
            Xtr = Xtr[:, feat_subset]; Xte = Xte[:, feat_subset]
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0)
        clf.fit(Xtr, y[tr])
        p = clf.predict(Xte)
        preds[te] = p
        accs.append(accuracy_score(y[te], p))
    return np.mean(accs), preds

print('\n== Held-out SEARCH tree accuracy by depth (all sources) ==')
for depth in [1,2,3,4,5,6,8,10,None]:
    acc, _ = cv_tree(X, y, fold, depth)
    print(f'depth={depth}: heldout acc={acc:.4f}')

# feature importance from full-depth tree trained on all
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=0)
clf.fit(X, y)
imp = sorted(zip(FEAT_NAMES, clf.feature_importances_), key=lambda x: -x[1])
print('\n== Top feature importances (depth6 full fit) ==')
for n, v in imp[:15]:
    print(f'  {n}: {v:.4f}')

np.save(_p(r"work\onto\_detC_X.npy"), X)
import pickle as pk
pk.dump({'FEAT_NAMES':FEAT_NAMES,'y':y,'fold':fold,'src':src}, open(_p(r"work\onto\_detC_meta.pkl"),'wb'))
