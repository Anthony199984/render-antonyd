"""Arma trabajos/ACTUAL.json: reparte los cuadros de todos los tramos en N trozos parejos (uno por máquina).
   python3 armar_trabajo.py plan.json N  > trabajos/ACTUAL.json
plan.json = {"nombre": "...", "fps": 15, "video_fps": 30,
             "tramos": [{"etiqueta": "CAMINAR", "t0": 0, "t1": 1.03, "repetir_ultimo": 7, "cfg": {...cfg de max_comp sin tiempos ni out...}}]}"""
import json, sys
plan = json.load(open(sys.argv[1])); N = int(sys.argv[2]); fps = plan.get('fps', 15)
cuadros = []                                   # (tramo, t)
for k, tr in enumerate(plan['tramos']):
    n = int(round((tr['t1'] - tr['t0']) * fps))
    cuadros += [(k, round(tr['t0'] + i / fps, 4)) for i in range(n)]
por = -(-len(cuadros) // N); trozos = []
for i in range(N):
    parte = cuadros[i * por:(i + 1) * por]
    if not parte: break
    runs = []
    for g, (k, t) in enumerate(parte, start=i * por):
        if runs and runs[-1]['tramo'] == k: runs[-1]['tiempos'].append(t)
        else: runs.append(dict(tramo=k, inicio=g, tiempos=[t]))
    trozos.append(dict(id=f'{i:02d}', corridas=runs))
plan['trozos'] = trozos; plan['total'] = len(cuadros)
plan['etiquetas'] = [plan['tramos'][k].get('etiqueta', '') for k, _ in cuadros]
print(json.dumps(plan, ensure_ascii=False))
