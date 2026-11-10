"""ZIP 40 · ESQUELETO v2: la COLUMNA SE DOBLA SUAVE (archivo nuevo: esqueleto_max.py del ZIP 39 queda igual).

En el ZIP 39 cada vértebra iba pegada entera a uno de los 7 huesos de Max de la espalda y el cuello (cadera, Spine1,
Spine2, pecho, cuello 1 y 2, cabeza): al agacharse o doblarse, la columna se doblaba en 4-5 tramos rectos, con escalones.
Ahora cada vértebra (y su disco) reparte el doblez entre los dos huesos de Max que tiene cerca, según dónde está: la
columna hace una curva pareja, vértebra por vértebra. Las costillas flotantes y las de abajo (8.ª a 12.ª) siguen a su
vértebra; la caja del pecho (costillas 1.ª a 7.ª, esternón y cartílagos) sigue rígida con el pecho, como en el cuerpo.

Se usa igual que EsqueletoMax (mismo .npz, mismas funciones):
    from soma.esqueleto_max_v2 import EsqueletoMaxV2
    esq = EsqueletoMaxV2(m2)
    for clave, (V, N) in esq.piezas().items(): ...       # sube cada pieza una vez (clave: hueso de Max o vértebra)
    mods = esq.modelos_solos(clip, t)                    # {clave: matriz 4x4}
"""
import numpy as np

from .esqueleto_max import EsqueletoMax

CADENA = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head')
ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
       'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12}


class EsqueletoMaxV2(EsqueletoMax):
    COLUMNA_SUAVE = True
    COSTILLA_DESDE = 8          # desde esta costilla, cada costilla sigue a su vértebra (las de arriba, al pecho)

    def __init__(s, m, archivo=None):
        super().__init__(m, archivo)
        s._mezcla = {}                                   # i (índice de hueso) -> (j_a, j_b, peso_b, centro)
        if not s.COLUMNA_SUAVE:
            return
        ix = m.idx
        cad = [(ix[n], float(m.pA[n][1])) for n in CADENA if n in ix]
        mitad = [(cad[k][1] + cad[k + 1][1]) / 2 for k in range(len(cad) - 1)] + [cad[-1][1] + 0.05]
        def pesos(y):
            if y <= mitad[0]:
                return cad[0][0], cad[0][0], 0.0
            for k in range(len(mitad) - 1):
                if mitad[k] <= y <= mitad[k + 1]:
                    w = (y - mitad[k]) / (mitad[k + 1] - mitad[k])
                    w = w * w * (3 - 2 * w)
                    return cad[k][0], cad[k + 1][0], float(w)
            return cad[-1][0], cad[-1][0], 0.0
        centros = {n: s.V0[a:b].mean(0) for n, (a, b, j) in zip(s.nombres, s.rangos)}
        def es_columna(n):
            return ('vertebra' in n or n in ('Atlas', 'Axis') or n.startswith('Intervertebral disk')) and 'Sacrum' not in n
        vert_de = {}
        for n in s.nombres:
            if 'thoracic vertebra' in n and not n.startswith('Intervertebral'):
                vert_de[ORD[n.split()[0].lower()]] = n
        for i, n in enumerate(s.nombres):
            if es_columna(n):
                s._mezcla[i] = pesos(centros[n][1]) + (centros[n],)
            elif n.endswith(' rib') and n.split()[0] in ('Left', 'Right'):
                k = ORD.get(n.split()[1].lower(), 0)
                if k >= s.COSTILLA_DESDE and k in vert_de:
                    v = vert_de[k]
                    s._mezcla[i] = pesos(centros[v][1]) + (centros[v],)

    def piezas(s):
        if getattr(s, '_piezas2', None) is None:
            s._piezas2 = {}
            Fj = s.J[s.F[:, 0]]
            hueso = np.zeros(len(s.V0), np.int64)
            for i, (a, b, j) in enumerate(s.rangos):
                hueso[a:b] = i
            Fh = hueso[s.F[:, 0]]
            mezclado = np.isin(Fh, list(s._mezcla))
            for j in np.unique(Fj[~mezclado]):
                tri = s.F[(Fj == j) & ~mezclado].reshape(-1)
                s._piezas2[int(j)] = (s.V0[tri], s._nrm(s.N0[tri]))
            for i in s._mezcla:
                tri = s.F[Fh == i].reshape(-1)
                s._piezas2[f'v{i}'] = (s.V0[tri], s._nrm(s.N0[tri]))
        return s._piezas2

    @staticmethod
    def _nrm(N):
        return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

    def modelos(s, G=None, Pm=None, b=None, mover=None):
        from scipy.spatial.transform import Rotation as Ro, Slerp
        if G is None:
            G, Pm, b = s._ult
        K = s._K(G, Pm, b)
        Cz = np.eye(4); Cz[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0.]]
        S = np.eye(4) * s.m.base.scale; S[3, 3] = 1; S[:3, 3] = -np.array([0, 0, s.m.base.zoff]) * s.m.base.scale
        if mover is not None:
            S[:3, 3] += np.asarray(mover, float)
        out = {}
        for clave in s.piezas():
            if isinstance(clave, (int, np.integer)):
                out[clave] = (S @ Cz @ K[clave]).astype(np.float32)
        for i, (ja, jb, w, c) in s._mezcla.items():
            Ka, Kb = K[ja], K[jb]
            if w <= 1e-4 or ja == jb:
                M = Ka
            else:
                R = Slerp([0, 1], Ro.from_matrix(np.stack([Ka[:3, :3], Kb[:3, :3]])))([w]).as_matrix()[0]
                p = (1 - w) * (Ka[:3, :3] @ c + Ka[:3, 3]) + w * (Kb[:3, :3] @ c + Kb[:3, 3])
                M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p - R @ c
            out[f'v{i}'] = (S @ Cz @ M).astype(np.float32)
        return out
