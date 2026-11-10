"""Renderiza un trozo: python3 trozo.py trabajos/ACTUAL.json <id> <carpeta_salida>. Cada corrida es un proceso nuevo (sin fugas)."""
import json, os, sys, subprocess, shutil, time, glob
job = json.load(open(sys.argv[1])); tid = sys.argv[2]; sal = sys.argv[3]; os.makedirs(sal, exist_ok=True)
tz = next(t for t in job['trozos'] if t['id'] == tid); t0 = time.time()
for r in tz['corridas']:
    tr = job['tramos'][r['tramo']]; tmp = os.path.abspath(f'_tmp_{tid}')
    shutil.rmtree(tmp, ignore_errors=True)
    cfg = dict(tr['cfg'], tiempos=r['tiempos'], out=tmp, tiempo_cam=tr['t0'])
    env = dict(os.environ); env.update(tr.get('env', {}))
    subprocess.run([sys.executable, job.get('script', '/home/claude/fisica/max_comp.py'), json.dumps(cfg)], check=True, env=env,
                   cwd=os.path.dirname(job.get('script', '/home/claude/fisica/max_comp.py')))
    fs = sorted(glob.glob(f'{tmp}/*.png'), key=lambda p: int(os.path.basename(p)[:-4]))
    for i, f in enumerate(fs): shutil.move(f, f"{sal}/{r['inicio'] + i:05d}.png")
print(f'trozo {tid}: {sum(len(r["tiempos"]) for r in tz["corridas"])} cuadros en {time.time() - t0:.0f} s')
open(f'{sal}/tiempo_{tid}.txt', 'w').write(f'{time.time() - t0:.1f}\n')
