import collections, re, math
from _detC_load import load, SEARCH, SEARCHS

recs = load()
srch = [d for d in recs if d['LABEL'] in SEARCHS]

tok_re = re.compile(r"[a-z_]+")
def toks(s):
    return set(tok_re.findall((s or '').lower()))

# per-class token counts on current_prompt
cls_tok = {c: collections.Counter() for c in SEARCH}
cls_n = collections.Counter()
glob_tok = collections.Counter()
for d in srch:
    t = toks(d['current_prompt'])
    c = d['LABEL']
    cls_n[c]+=1
    for w in t:
        cls_tok[c][w]+=1
        glob_tok[w]+=1

N = len(srch)
print('=== distinctive tokens per SEARCH class (log-odds vs rest, min freq 40) ===')
for c in SEARCH:
    scores=[]
    nc = cls_n[c]
    for w,gf in glob_tok.items():
        if gf < 40: continue
        inc = cls_tok[c][w]
        # p(w|c) vs p(w|not c)
        pc = (inc+1)/(nc+2)
        prest = (gf-inc+1)/(N-nc+2)
        lo = math.log(pc/prest)
        scores.append((lo, w, inc, gf))
    scores.sort(reverse=True)
    print(f'\n[{c}] n={nc}')
    for lo,w,inc,gf in scores[:12]:
        print(f'   {w:14s} logodds={lo:+.2f} in-class={inc}/{nc}={inc/nc:.2f} global={gf}')

# Now: can a token-presence classifier separate? quick check on a few strong action words
print('\n=== presence of explicit action words -> label ===')
probes = {
 'grep':['grep'], 'search':['search'], 'find':['find'], 'list':['list','ls'],
 'glob':['glob'], 'open':['open'], 'read':['read'], 'show':['show'],
 'star':['*'], 'files':['files'], 'directory':['directory','dir','folder'],
}
def has(d, ws):
    p=(d['current_prompt'] or '').lower()
    return any(w in p for w in ws)
for name,ws in probes.items():
    sub=[d for d in srch if has(d,ws)]
    if not sub: continue
    c=collections.Counter(d['LABEL'] for d in sub)
    tot=len(sub); top=c.most_common(1)[0]
    print(f"prompt~'{name}': n={tot:5d} purity={top[1]/tot:.3f}->{top[0]:14s} | "+
          ' '.join(f'{k}={v/tot:.2f}' for k,v in c.most_common()))
