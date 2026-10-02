import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, re, collections
import numpy as np

OOF = _p(r"work\onto\oof.pkl")
TRAIN = _p(r"data\data\train.jsonl")
FOLDS = _p(r"work\onto\folds.json")

oof = pickle.load(open(OOF, 'rb'))
lab = dict(zip(oof['ids'], oof['y']))
folds = json.load(open(FOLDS))
recs = [json.loads(l) for l in open(TRAIN, encoding='utf-8')]
for d in recs:
    d['LABEL'] = lab.get(d['id'])
    d['FOLD'] = folds.get(d['id'])

SEARCH = {'read_file', 'grep_search', 'glob_pattern', 'list_directory'}

# split id into au/sim + timestamp + step
def parse_id(i):
    # sess_sim_20260522_024730-step_08
    m = re.match(r'sess_(au|sim)_(\d{8})_(\d+)-step_(\d+)', i)
    if m:
        return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    return None, None, None, None

for d in recs:
    src, day, seq, step = parse_id(d['id'])
    d['SRC'] = src; d['DAY'] = day; d['SEQ'] = seq; d['STEP'] = step

# quick sanity: label dist within SEARCH
srch = [d for d in recs if d['LABEL'] in SEARCH]
print('total recs', len(recs), 'SEARCH recs', len(srch))
print('SEARCH label dist:', collections.Counter(d['LABEL'] for d in srch))
print('SRC dist all:', collections.Counter(d['SRC'] for d in recs))
print('SRC dist SEARCH:', collections.Counter(d['SRC'] for d in srch))
print()
# label dist by source within SEARCH
for src in ('au', 'sim'):
    sub = [d for d in srch if d['SRC'] == src]
    c = collections.Counter(d['LABEL'] for d in sub)
    tot = len(sub)
    print(f'SRC={src} n={tot}:', {k: round(v/tot, 3) for k, v in c.most_common()})
