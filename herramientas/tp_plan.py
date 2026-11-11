"""Arma trabajos/TP.json:  python3 tp_plan.py <nombre> <ancho> <fps> <cuadros_por_trozo> [tomas a incluir, p. ej. 1,2]"""
import sys, json, math
TOMAS = [(1, 0.0, 7.0), (2, 7.0, 12.3), (3, 12.3, 17.5), (4, 17.5, 27.95), (5, 27.95, 36.3), (6, 36.3, 43.7), (7, 43.7, 52.05),
         (8, 52.05, 61.3), (9, 61.3, 69.75), (10, 69.75, 73.36), (11, 73.36, 83.35), (12, 83.35, 89.2)]
nombre, ancho, fps, por = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
solo = set(int(x) for x in sys.argv[5].split(',')) if len(sys.argv) > 5 else None
trozos = []; total = 0
for n, a, b in TOMAS:
    k0, k1 = math.ceil(a * fps - 1e-9), math.ceil(b * fps - 1e-9)
    if n == 12: k1 = round(b * fps)
    total = max(total, k1)
    if solo and n not in solo: continue
    paso = k1 - k0 if n == 1 else por                     # la T1 va entera (el rebobinado usa sus propios cuadros)
    for i in range(k0, k1, paso):
        trozos.append(dict(id=f't{n:02d}_{len(trozos):02d}', toma=n, k0=i, k1=min(i + paso, k1)))
plan = dict(nombre=nombre, modo='render', fps=fps, ancho=ancho, res=f'{ancho}x{ancho * 16 // 9}', k_total=total if not solo else sum(t['k1'] - t['k0'] for t in trozos), trozos=trozos)
json.dump(plan, open('trabajos/TP.json', 'w'), indent=1)
print(len(trozos), 'trozos ·', plan['k_total'], 'cuadros')
