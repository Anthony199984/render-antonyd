"""Arma trabajos/SF.json para el video de superfuerza.

    python3 sf_plan.py medir  <nombre>                       -> prueba de costo (cuadros sueltos que se tiran)
    python3 sf_plan.py render <nombre> medicion.json N P     -> el render de verdad en N máquinas, P procesos por máquina
"""
import sys, json, math

FPS = 60; K = int(round(74.333 * FPS))                                   # 4460 cuadros
TRAMOS = [0, 13.16, 21.41, 34.73, 44.39, 53.10, 69.10, 74.333]
KT = [int(round(t * FPS)) for t in TRAMOS]; KT[-1] = K
INTERIORES = [(8.27, 13.16, 'brazo'), (44.39, 51.0, 'carcel'), (51.0, 52.60, 'tendon'), (58.32, 62.59, 'corazon'), (65.73, 67.95, 'pareja')]
# cuadros JPG que tiene cada vista por dentro (render_final.sh)
N_DENTRO = {'brazo': 293, 'tendon': 96, 'carcel': 397, 'corazon': 257, 'pareja': 133}


def interior(k):
    t = k / FPS
    for a, b, n in INTERIORES:
        if a <= t < b: return n, min(int(round((t - a) * FPS)), int(round((b - a) * FPS)) - 1)
    return None


def tramo(k): return max(i for i in range(7) if KT[i] <= k)


ESCENA = [k for k in range(K) if interior(k) is None]


def plan_medir(nombre):
    trozos = []
    for i in range(7):
        ks = [k for k in ESCENA if KT[i] <= k < KT[i + 1]]
        m = ks[::45] + [ks[-1]]                                          # un cuadro cada 0,75 s de video
        trozos.append({'id': f'm{i}', 'tareas': [{'tipo': 'escena', 'cuadros': m}], 'paralelo': 1})
    ks = [k for k in ESCENA if KT[2] <= k < KT[3]][100:]                 # prueba de procesos en paralelo (cocina)
    trozos.append({'id': 'mpar3', 'tareas': [{'tipo': 'escena', 'cuadros': ks[j * 40:j * 40 + 4]} for j in range(3)], 'paralelo': 3})
    trozos.append({'id': 'mpar2', 'tareas': [{'tipo': 'escena', 'cuadros': ks[200 + j * 40:200 + j * 40 + 4]} for j in range(2)], 'paralelo': 2})
    for n, N in N_DENTRO.items():
        trozos.append({'id': f'd{n}', 'tareas': [{'tipo': f'dentro:{n}', 'cuadros': sorted({0, N // 3, 2 * N // 3, N - 1})}], 'paralelo': 1})
    return {'nombre': nombre, 'modo': 'medir', 'k_total': K, 'trozos': trozos}


if __name__ == '__main__':
    modo, nombre = sys.argv[1], sys.argv[2]
    if modo == 'medir':
        p = plan_medir(nombre)
        json.dump(p, open('trabajos/SF.json', 'w'), indent=1)
        print(len(p['trozos']), 'máquinas de medición;', sum(len(t['cuadros']) for x in p['trozos'] for t in x['tareas']), 'cuadros sueltos')


# ============================================================================== reparto por costo (con la medición)
def modelo(medicion):
    M = json.load(open(medicion))
    esc = {}; ini_esc = {}; den = {}; ini_den = {}
    for m in M:
        if m['maquina'].startswith('trozo-mpar'): continue
        c = sorted(m['cuadros'])
        if m['tipo'] == 'escena':
            i = tramo(c[0][0]); fijo = c[1][1] if len(c) > 1 else c[0][1]
            ini_esc[i] = m['arranque'] + max(c[0][1] - fijo, 0)
            esc[i] = [(c[0][0], fijo)] + [(k, w) for k, w, _ in c[1:]]
        else:
            n = m['tipo'].split(':')[1]; ini_den[n] = m['arranque']; den[n] = [(f, w) for f, w, _ in c]

    def interp(pts, x):
        if x <= pts[0][0]: return pts[0][1]
        for (a, wa), (b, wb) in zip(pts, pts[1:]):
            if a <= x <= b: return wa + (wb - wa) * (x - a) / max(b - a, 1)
        return pts[-1][1]
    costo = []
    for k in range(K):
        d = interior(k)
        costo.append(0.05 + interp(den[d[0]], d[1]) if d else interp(esc[tramo(k)], k))
    return costo, ini_esc, ini_den


def partir(costo, ini_esc, ini_den, T, FIJO=8.0):
    """Corta el video en pedazos seguidos que no pasen de T segundos estimados. Devuelve [(k0, k1, costo)]."""
    trozos = []; k = 0
    while k < K:
        a = k; c = FIJO; vistos = set()
        while k < K:
            d = interior(k); clave = ('d', d[0]) if d else ('e', tramo(k))
            extra = costo[k] + (0 if clave in vistos else (ini_den[clave[1]] if d else ini_esc[clave[1]]))
            if c + extra > T and k > a: break
            c += extra; vistos.add(clave); k += 1
        trozos.append((a, k, c))
    return trozos


def tareas(k0, k1):
    """Procesos de una máquina, en orden de video: escenas por tramo y vistas por dentro."""
    out = []
    for k in range(k0, k1):
        d = interior(k); tipo = f'dentro:{d[0]}' if d else 'escena'; x = d[1] if d else k
        if out and out[-1]['tipo'] == tipo and (tipo != 'escena' or tramo(out[-1]['cuadros'][-1]) == tramo(k)):
            if x not in out[-1]['cuadros']: out[-1]['cuadros'].append(x)
        else: out.append({'tipo': tipo, 'cuadros': [x]})
    return out


def calibrar(costo, plan_viejo, informe):
    """Corrige el costo con lo que de verdad tardó cada máquina en un render anterior (real / estimado)."""
    import re
    P = json.load(open(plan_viejo)); real = {m.group(1): float(m.group(2)) for m in re.finditer(r'máquina (\w+): cuadros .*? render (\d+) s', open(informe).read())}
    for t in P['trozos']:
        if t['id'] in real:
            f = real[t['id']] / t['estimado_s']
            for k in range(t['k0'], t['k1']): costo[k] *= f
    return costo


def plan_render(nombre, medicion, N, plan_viejo=None, informe=None):
    costo, ie, idn = modelo(medicion)
    if plan_viejo: costo = calibrar(costo, plan_viejo, informe)
    lo, hi = 1.0, sum(costo) + 1000
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(partir(costo, ie, idn, mid)) <= N: hi = mid
        else: lo = mid
    tz = partir(costo, ie, idn, hi)
    trozos = [{'id': f'{i:02d}', 'k0': a, 'k1': b, 'estimado_s': round(c), 'tareas': tareas(a, b), 'paralelo': 1} for i, (a, b, c) in enumerate(tz)]
    return {'nombre': nombre, 'modo': 'render', 'k_total': K, 'trozos': trozos}, sum(costo)


if __name__ == '__main__' and sys.argv[1] == 'render':
    p, tot = plan_render(sys.argv[2], sys.argv[3], int(sys.argv[4]), *(sys.argv[6:8] if len(sys.argv) > 7 else ()))
    A = int(sys.argv[5]) if len(sys.argv) > 5 else 1080; H = A * 16 // 9; p['ancho'] = A; p['res'] = f'{A}x{H}'
    assert sum(t['k1'] - t['k0'] for t in p['trozos']) == K and p['trozos'][0]['k0'] == 0 and p['trozos'][-1]['k1'] == K
    json.dump(p, open('trabajos/SF.json', 'w'))
    print(f"costo total estimado {tot / 60:.1f} min de máquina · {len(p['trozos'])} máquinas")
    for t in p['trozos']:
        print(t['id'], t['k0'], t['k1'], t['estimado_s'], 's', [(x['tipo'], len(x['cuadros'])) for x in t['tareas']])
