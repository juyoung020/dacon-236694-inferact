import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
def _p(rel):  # 저장소 루트 기준 경로
    return _os.path.join(_ROOT, *[x for x in rel.split('\\') if x])
import json, pickle, re, collections
import numpy as np

OOF = _p(r"work\onto\oof.pkl")
TRAIN = _p(r"data\data\train.jsonl")
FOLDS = _p(r"work\onto\folds.json")

SEARCH = ['read_file', 'grep_search', 'glob_pattern', 'list_directory']
SEARCHS = set(SEARCH)

def load():
    oof = pickle.load(open(OOF, 'rb'))
    lab = dict(zip(oof['ids'], oof['y']))
    folds = json.load(open(FOLDS))
    recs = [json.loads(l) for l in open(TRAIN, encoding='utf-8')]
    for d in recs:
        d['LABEL'] = lab.get(d['id'])
        d['FOLD'] = folds.get(d['id'])
        i = d['id']
        m = re.match(r'sess_(au|sim)_(\d+)_(\d+)-step_(\d+)', i)
        if m:
            d['SRC'] = m.group(1)
            d['SEQ'] = m.group(2)      # timestamp-ish for sim, id for au
            d['SUB'] = int(m.group(3))
            d['STEP'] = int(m.group(4))
        else:
            d['SRC'] = None; d['SEQ'] = None; d['SUB'] = None; d['STEP'] = None
    return recs

if __name__ == '__main__':
    recs = load()
    srch = [d for d in recs if d['LABEL'] in SEARCHS]
    print('SEARCH recs', len(srch))
    for src in ('au', 'sim'):
        sub = [d for d in srch if d['SRC'] == src]
        c = collections.Counter(d['LABEL'] for d in sub)
        tot = len(sub)
        print(f'SRC={src} n={tot}:', {k: round(v/tot, 3) for k, v in c.most_common()})
    # overall all-labels by src
    print()
    for src in ('au', 'sim'):
        sub = [d for d in recs if d['SRC'] == src]
        c = collections.Counter(d['LABEL'] for d in sub)
        tot = len(sub)
        print(f'ALL SRC={src} n={tot}:', {k: round(v/tot, 3) for k, v in c.most_common()})
