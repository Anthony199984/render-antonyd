"""ZIP 39 · ESQUELETO HUMANO REAL DENTRO DE MAX (para el video del esqueleto que sale del cuerpo).

El esqueleto es BodyParts3D 4.0 (Database Center for Life Science, Japón), un hombre adulto real escaneado, licencia
CC BY 4.0 (pide crédito: "BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution 4.0").
264 huesos (cráneo, mandíbula y dientes, columna con discos, costillas y cartílagos, esternón, pelvis, brazos, manos con
todas las falanges, piernas, rótulas y pies). Está ajustado DENTRO de Max (su altura, cadera, hombros y cada tramo de
brazos, piernas, manos y dedos) y cada hueso va pegado a su hueso de Max, así que hace cualquier animación de Max.

    from soma.esqueleto_max import EsqueletoMax
    esq = EsqueletoMax(m)                 # m = MaxCaraV14 (o cualquier Max con pose_mats)
    res = esq.pose(clip, t)               # dibuja a Max en t y devuelve el esqueleto: (V, N, F) en el mundo de Max
    # o, si ya se llamó m.pose(clip, t):  esq.malla()
    # para que el esqueleto haga OTRA animación que el cuerpo: esq.pose(otro_clip, t, mover=(dx, dy, 0))
Colores: EsqueletoMax.COLOR (hueso), COLOR_DIENTES.
"""
import os
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))


class EsqueletoMax:
    COLOR = (0.86, 0.80, 0.68)          # hueso (marfil, algo amarillento)
    COLOR_DIENTES = (0.93, 0.91, 0.84)
    ARCHIVO = os.path.join(AQUI, 'esqueleto_max_v5.npz')

    def __init__(s, m, archivo=None):
        d = np.load(archivo or s.ARCHIVO, allow_pickle=False)
        s.m = m
        s.V0 = d['V'].astype(float); s.N0 = d['N'].astype(float); s.F = d['F'].astype(np.int64)
        s.J = d['J'].astype(np.int64); s.nombres = [str(x) for x in d['nombres']]; s.rangos = d['rangos']
        dientes = np.zeros(len(s.V0), bool)
        for n, (a, b, j) in zip(s.nombres, s.rangos):
            if 'tooth' in n:
                dientes[a:b] = True
        s.dientes = dientes
        s.fuente = str(d['fuente'])
        # se guarda la pose con que se dibujó a Max (para poner el esqueleto en la misma pose)
        s._ult = None
        orig = m.pose_mats

        def pose_mats(G, Pm, b, apply=True):
            s._ult = (G, Pm, b)
            return orig(G, Pm, b, apply)
        m.pose_mats = pose_mats

    def _K(s, G, Pm, b):
        from .rig import _M
        return np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])

    def pose(s, clip, t, bucle=False, mover=None):
        """Pone a Max en t (como m.pose) y devuelve el esqueleto en esa pose. Lo de Max queda en s.res_max."""
        s.res_max = s.m.pose(clip, t, bucle)
        G, Pm, b = s._ult
        return s.malla(G, Pm, b, mover)

    def pose_sola(s, clip, t, bucle=False, mover=None):
        """Solo el esqueleto (sin dibujar a Max): para un esqueleto que hace otra animación."""
        G, Pm, b = s.m.globales(clip, t, bucle)
        return s.malla(G, Pm, b, mover)

    # ---- por hueso (cada pedazo es rígido): se sube una vez y en cada cuadro solo cambia su matriz (mucho más rápido)
    def piezas(s):
        """{hueso de Max: (V, N) triángulos sueltos en la pose de enlace}. Se sube una vez al render."""
        if getattr(s, '_piezas', None) is None:
            s._piezas = {}
            Fj = s.J[s.F[:, 0]]
            for j in np.unique(Fj):
                tri = s.F[Fj == j].reshape(-1)
                s._piezas[int(j)] = (s.V0[tri], s.N0[tri] / np.maximum(np.linalg.norm(s.N0[tri], axis=1, keepdims=True), 1e-9))
        return s._piezas

    def modelos(s, G=None, Pm=None, b=None, mover=None):
        """{hueso de Max: matriz 4x4} que lleva cada pieza de la pose de enlace al mundo de Max en esta pose."""
        if G is None:
            G, Pm, b = s._ult
        K = s._K(G, Pm, b)
        Cz = np.eye(4); Cz[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0.]]
        S = np.eye(4) * s.m.base.scale; S[3, 3] = 1; S[:3, 3] = -np.array([0, 0, s.m.base.zoff]) * s.m.base.scale
        if mover is not None:
            S[:3, 3] += np.asarray(mover, float)
        return {j: (S @ Cz @ K[j]).astype(np.float32) for j in s.piezas()}

    def modelos_solos(s, clip, t, bucle=False, mover=None):
        G, Pm, b = s.m.globales(clip, t, bucle)
        s._ult = (G, Pm, b)
        return s.modelos(G, Pm, b, mover)

    def malla(s, G=None, Pm=None, b=None, mover=None):
        if G is None:
            if s._ult is None:
                raise ValueError('primero m.pose(clip, t) o usar esq.pose(clip, t)')
            G, Pm, b = s._ult
        K = s._K(G, Pm, b)
        Kv = K[s.J]
        V = np.einsum('vij,vj->vi', Kv[:, :3, :3], s.V0) + Kv[:, :3, 3]
        N = np.einsum('vij,vj->vi', Kv[:, :3, :3], s.N0)
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        V = V @ Cz.T; N = N @ Cz.T
        V = (V - np.array([0, 0, s.m.base.zoff])) * s.m.base.scale
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        if mover is not None:
            V = V + np.asarray(mover, float)
        return V, N, s.F
