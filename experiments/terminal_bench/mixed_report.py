#!/usr/bin/env python3
"""One-shot status report for the three mixed-pool arms (GV/LAV-luna/LAV-ds).
Reads /tmp/all_gv.txt (concatenated GV logs), /tmp/inflight.txt (log:running
lines), and the lav_*.jsonl files listed on argv."""
import re, sys, json, glob, os, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, '../vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash')
truth = {}
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p): continue
    rs = [json.load(open(f)).get('reward', 0) for f in sorted(glob.glob(p + '/*_trajectory.json'))]
    truth[d] = sum(1 for r in rs if r)
mixed = {t for t, o in truth.items() if 0 < o < 5}
grp = {}
cur = None
for l in open(os.path.join(HERE, 'official_groups.txt')):
    l = l.strip()
    if l.startswith('# group'): cur = int(l.split()[2])
    elif l and cur: grp[l] = cur

best = {}
for l in open('/tmp/all_gv.txt'):
    m = re.search(r'\] (\S+)/(\S+) reward=(\d\.\d) -> (\w+) score=([\d.]+)', l)
    if m and m.group(1) in mixed:
        best[(m.group(1), m.group(2))] = (float(m.group(5)), float(m.group(3)), m.group(4))
by = collections.defaultdict(list)
for (t, tr), v in best.items(): by[t].append(v)

gtot = collections.Counter(); gj = collections.Counter()
for t in mixed: gtot[grp[t]] += 5
for (t, tr) in best: gj[grp[t]] += 1
print('== ① GV(Luna) mixed ==')
print('每组判决/总数:', '  '.join(f'g{k}:{gj[k]}/{gtot[k]}' for k in range(1, 7)),
      f'| 合计 {sum(gj.values())}/180')
try:
    infl = open('/tmp/inflight.txt').read()
    print('在跑:', infl.strip())
except FileNotFoundError:
    pass
comp = sorted(t for t in by if len(by[t]) >= 5)
ok = [t for t in comp if sorted(by[t], reverse=True)[0][1] == 1.0]
print(f'BoN: 判满 {len(comp)}/36 池, 对 {len(ok)} | MISS: {[t for t in comp if t not in ok]}')
v1 = collections.Counter(); v0 = collections.Counter(); s1 = []; s0 = []
for (t, tr), (s, r, v) in best.items():
    (v1 if r == 1.0 else v0)[v] += 1
    (s1 if r == 1.0 else s0).append(s)
print('判决分布 r=1:', dict(v1))
print('判决分布 r=0:', dict(v0))
def hist(xs):
    b = collections.Counter(min(int(x * 10), 9) for x in xs)
    return ' '.join(f'{k/10:.1f}:{b[k]}' for k in sorted(b))
if s1: print(f'score r=1: n={len(s1)} med={statistics.median(s1):.2f} | {hist(s1)}')
if s0: print(f'score r=0: n={len(s0)} med={statistics.median(s0):.2f} | {hist(s0)}')

def lav_report(name, files):
    rows = []
    for f in files:
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r['task'] in mixed]
    print(f'== {name} ==')
    if not rows:
        print('  暂无结果'); return
    okc = sum(1 for r in rows if r['picked_reward'] == 1.0)
    allsc = [x for r in rows for x in r['scores']]
    mid = sum(1 for x in allsc if 0.45 <= x <= 0.55)
    print(f'  任务完成 {len(rows)}/36 | BoN {okc}/{len(rows)}')
    print(f'  score: n={len(allsc)} 落0.45-0.55: {mid} ({100*mid/len(allsc):.0f}%) '
          f'范围[{min(allsc):.2f},{max(allsc):.2f}] med={statistics.median(allsc):.2f}')
    miss = [r['task'] for r in rows if r['picked_reward'] != 1.0]
    print(f'  错选: {miss}')

lav_report('② LAV(Luna)', sys.argv[1].split(',') if len(sys.argv) > 1 else [])
lav_report('③ LAV(deepseek)', sys.argv[2].split(',') if len(sys.argv) > 2 else [])
