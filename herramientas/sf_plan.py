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
