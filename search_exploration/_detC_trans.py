import collections, re
from _detC_load import load, SEARCH, SEARCHS

recs = load()
srch = [d for d in recs if d['LABEL'] in SEARCHS]

def last_action(d):
    acts = [h for h in d['history'] if h.get('role')=='assistant_action']
    return acts[-1] if acts else None

# fine-grained result category
def rescat(a):
    if a is None: return ('NONE','')
    rs = (a.get('result_summary') or '').lower()
    name = a['name']
    # extract match count for grep
    if name=='grep_search':
        m = re.search(r'(\d+)\s+match', rs)
        if 'no match' in rs or '0 match' in rs: return ('grep','0match')
        if m:
            n=int(m.group(1))
            if n==0: return ('grep','0match')
            return ('grep','matches')
        return ('grep','other')
    if name=='glob_pattern':
        m = re.search(r'(\d+)', rs)
        if 'no ' in rs or '0 ' in rs: return ('glob','0')
        return ('glob','some')
    if name=='list_directory':
        return ('list','ok')
    if name=='read_file':
        return ('read','ok')
    if name in ('edit_file','apply_patch','write_file'):
        return ('edit', 'ok' if ('ok' in rs or 'patch' in rs) else 'other')
    if name in ('run_tests','lint_or_typecheck'):
        if 'pass' in rs: return (name,'pass')
        if 'fail' in rs: return (name,'fail')
        return (name,'other')
    if name=='run_bash':
        return ('run_bash','fail' if 'fail' in rs or 'error' in rs else 'ok')
    return (name,'other')

print('=== transition: (last_action, last_result) -> next SEARCH label ===')
tab = collections.defaultdict(collections.Counter)
for d in srch:
    key = rescat(last_action(d))
    tab[key][d['LABEL']] += 1
rows = sorted(tab.items(), key=lambda kv: -sum(kv[1].values()))
for key, c in rows:
    tot = sum(c.values())
    if tot < 80: continue
    top = c.most_common(1)[0]
    print(f'{str(key):30s} n={tot:5d} purity={top[1]/tot:.3f}->{top[0]:14s} | '+
          ' '.join(f'{k}={v/tot:.2f}' for k,v in c.most_common()))
