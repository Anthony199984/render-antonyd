"""RENDER EN GITHUB CON EL MAX NUEVO · paso 2: el plan (trabajos/MN.json). Al subirlo al repo, arranca solo.
    python3 mn_plan.py NOMBRE MAQUINAS [--medir] [--de K0 --a K1] [--solo id,id]
Usa /tmp/mn_proyecto.json (de mn_proyecto.py). Reparte los cuadros en pedazos seguidos; si hay una medición anterior
(trabajos/medicion_mn.json: [[k, segundos], ...]) reparte por costo para que todas las máquinas terminen a la vez.
--medir: cada máquina dibuja unos cuadros sueltos para medir (no sale video).  --solo: rehacer solo esos pedazos."""
import sys, os, json, numpy as np
a = sys.argv[1:]; NOMBRE, MAQ = a[0], int(a[1]); P = json.load(open('/tmp/mn_proyecto.json'))
def opt(n, d=None): return a[a.index(n) + 1] if n in a else d
k0, k1 = int(opt('--de', 0)), int(opt('--a', P['cuadros']))
plan = dict(nombre=NOMBRE, video=P['video'], ancho=P['ancho'], alto=P['alto'], fps=round(P['fps'], 3), sha=P['sha'], crf=18)
if '--medir' in a:
    ks = np.linspace(k0, k1 - 1, MAQ * 3).astype(int).tolist()
    plan.update(modo='medir', trozos=[dict(id=f'm{i:02d}', cuadros=ks[i::MAQ]) for i in range(MAQ)])
else:
    med = 'trabajos/medicion_mn.json'
    c = np.ones(k1 - k0)
    if os.path.exists(med):
        m = np.array(json.load(open(med))); c = np.interp(np.arange(k0, k1), m[:, 0], m[:, 1])
    acu = np.cumsum(c); cortes = [k0] + [k0 + int(np.searchsorted(acu, acu[-1] * i / MAQ)) for i in range(1, MAQ)] + [k1]
    plan.update(modo='render', trozos=[dict(id=f't{i:02d}', k0=int(cortes[i]), k1=int(cortes[i + 1])) for i in range(MAQ) if cortes[i + 1] > cortes[i]])
    if opt('--solo'): plan['trozos'] = [t for t in plan['trozos'] if t['id'] in opt('--solo').split(',')]
os.makedirs('trabajos', exist_ok=True); json.dump(plan, open('trabajos/MN.json', 'w'), indent=1)
print(f"plan {NOMBRE}: {plan['modo']}, {len(plan['trozos'])} máquinas, cuadros {k0}-{k1 - 1}, {plan['ancho']}x{plan['alto']}")
