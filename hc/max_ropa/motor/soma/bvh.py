"""Lector de BVH de Kimodo (esqueleto SOMA77, también sirve para cualquier BVH).

Kimodo exporta: Root (envoltorio en el origen) -> Hips (6 canales, posición en cm) -> resto (3 canales ZYX).
El reposo del BVH son los OFFSET con giro cero. Kimodo por defecto usa el reposo del dataset BONES-SEED,
no la T-pose estándar: por eso el reposo se lee de cada archivo y nunca se da por supuesto.

Clip(path).globales(t) -> (G[J,3,3], P[J,3]) rotaciones globales relativas al reposo y posiciones en metros (Y arriba).
"""
import numpy as np

def _rx(a):
    c, s = np.cos(a), np.sin(a); z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)
def _ry(a):
    c, s = np.cos(a), np.sin(a); z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)
def _rz(a):
    c, s = np.cos(a), np.sin(a); z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)
_ROT = {'X': _rx, 'Y': _ry, 'Z': _rz}

def _slerp_m(A, B, u):
    """Interpola matrices de giro (por lotes) vía cuaterniones."""
    from scipy.spatial.transform import Rotation as R
    qa, qb = R.from_matrix(A).as_quat(), R.from_matrix(B).as_quat()
    d = (qa*qb).sum(-1, keepdims=True); qb = np.where(d < 0, -qb, qb)
    q = qa*(1 - u) + qb*u; q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return R.from_quat(q).as_matrix()

class Clip:
    def __init__(s, path, escala=0.01):
        s.path = path
        txt = open(path, encoding='utf-8', errors='replace').read()
        hier, mot = txt.split('MOTION', 1)
        s.nombres, s.padres, off, chans = [], [], [], []
        pila = []            # índices de articulación abiertos; -2 = bloque End Site (se ignora)
        nuevo = None         # lo que abre el próximo '{'
        for line in hier.split('\n'):
            l = line.split()
            if not l: continue
            if l[0] in ('ROOT', 'JOINT'):
                padre = next((x for x in reversed(pila) if x >= 0), -1)
                s.nombres.append(l[1]); s.padres.append(padre); off.append(np.zeros(3)); chans.append([])
                nuevo = len(s.nombres) - 1
            elif l[0] == 'End': nuevo = -2
            elif l[0] == '{': pila.append(nuevo); nuevo = None
            elif l[0] == '}': pila.pop()
            elif l[0] == 'OFFSET' and pila and pila[-1] >= 0: off[pila[-1]] = np.array(list(map(float, l[1:4])))
            elif l[0] == 'CHANNELS' and pila and pila[-1] >= 0: chans[pila[-1]] = l[2:2 + int(l[1])]
        s.offsets = np.array(off)*escala
        s.canales = chans
        lm = [x for x in mot.split('\n') if x.strip()]
        nf = int(lm[0].split(':')[1]); s.dt = float(lm[1].split(':')[1]); s.fps = 1.0/s.dt
        datos = np.array([list(map(float, x.split())) for x in lm[2:2 + nf]])
        s.n = len(datos); s.dur = (s.n - 1)*s.dt
        J = len(s.nombres); s.idx = {n: i for i, n in enumerate(s.nombres)}
        L = np.tile(np.eye(3), (s.n, J, 1, 1)); T = np.tile(s.offsets, (s.n, 1, 1)).copy()
        c = 0
        for j in range(J):
            R = np.tile(np.eye(3), (s.n, 1, 1)); pos = s.offsets[j].copy()*np.ones((s.n, 3))
            for ch in s.canales[j]:
                v = datos[:, c]; c += 1
                if ch.endswith('rotation'): R = R @ _ROT[ch[0]](np.radians(v))
                else: pos[:, 'XYZ'.index(ch[0])] = v*escala
            L[:, j] = R; T[:, j] = pos
        s.L, s.T = L, T
        # FK de todo el clip
        s.G = np.zeros_like(L); s.P = np.zeros((s.n, J, 3))
        for j in range(J):
            p = s.padres[j]
            if p < 0: s.G[:, j] = L[:, j]; s.P[:, j] = T[:, j]
            else:
                s.G[:, j] = s.G[:, p] @ L[:, j]; s.P[:, j] = s.P[:, p] + np.einsum('nij,nj->ni', s.G[:, p], T[:, j])
        # posiciones de reposo (giros cero)
        s.reposo = np.zeros((J, 3))
        for j in range(J):
            p = s.padres[j]; s.reposo[j] = s.offsets[j] + (s.reposo[p] if p >= 0 else 0)

    def globales(s, t, bucle=False):
        """Rotaciones globales (relativas al reposo) y posiciones en el tiempo t (s), interpoladas."""
        f = t*s.fps
        if bucle and s.n > 1: f %= (s.n - 1)
        f = min(max(f, 0), s.n - 1); a = int(np.floor(f)); b = min(a + 1, s.n - 1); u = f - a
        if u < 1e-6 or a == b: return s.G[a], s.P[a]
        return _slerp_m(s.G[a], s.G[b], u), s.P[a]*(1 - u) + s.P[b]*u
