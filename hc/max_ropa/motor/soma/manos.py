"""Mano NATURAL de Max (media abierta) para cuando la animación no controla los dedos.

Kimodo solo genera 30 articulaciones: los dedos venían con la "mano relajada" de SOMA, que en Max se veía como una
paleta (dedos casi rectos 5-20°, pegados unos a otros, todos con la misma curva). Una mano relajada real:
- se cierra un poco más de índice a meñique (el meñique es el más doblado);
- los dedos se abren en abanico (índice hacia el pulgar, meñique hacia afuera);
- el pulgar queda suelto al lado del índice, sin pegarse.
Se aplica en exportar_bvh a toda mano cuyos dedos no se tocaron (si un agarre puso los dedos, se respetan).

    import manos; L77 = manos.aplicar(L77)          # (F,77,3,3) locales de Kimodo
"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/motionbricks_cpu')
from scipy.spatial.transform import Rotation as Rot

# grados de cierre por articulación (nudillo, medio, punta) y abanico en el nudillo (+ = hacia el pulgar)
CIERRE = {'Index': (14, 22, 10), 'Middle': (18, 28, 13), 'Ring': (22, 32, 15), 'Pinky': (27, 36, 17)}
ABANICO = {'Index': 7.0, 'Middle': 1.5, 'Ring': -4.0, 'Pinky': -10.0}
PULGAR = (12, 10, 8)            # se suma a la pose relajada de SOMA del pulgar: suelto, un poco hacia la palma

def _datos():
    import agarres_ik as A, exportar_bvh as xb
    return A, xb

def pose_mano(lado, escala=1.0):
    """{articulación: rotación local} de la mano natural. lado: 'izq' | 'der'. escala: 0 = recta, 1 = natural, >1 = más cerrada."""
    A, xb = _datos()
    s = A.LADOS[lado]; d0, n0 = s['d0'], s['n0']
    eje_curl = np.cross(d0, n0)
    t0 = np.cross(n0, d0) if lado == 'izq' else np.cross(d0, n0)          # lado del pulgar en T-pose
    # el abanico gira alrededor de la normal de la palma; signo elegido para que + mueva la punta hacia el pulgar
    Rp = Rot.from_rotvec(n0*np.radians(10)).as_matrix()
    sg = 1.0 if (Rp @ d0) @ t0 > 0 else -1.0
    out = {}
    for dedo, angs in CIERRE.items():
        for k, a in zip((2, 3, 4), angs):
            R = Rot.from_rotvec(eje_curl*np.radians(a*escala)).as_matrix()
            if k == 2: R = Rot.from_rotvec(n0*np.radians(sg*ABANICO[dedo])).as_matrix() @ R
            out[f"{s['prefijo']}{dedo}{k}"] = R
    eje_pulgar = np.cross(d0, n0 - 0.5*t0); eje_pulgar /= np.linalg.norm(eje_pulgar)
    for k, a in zip((1, 2, 3), PULGAR):
        j = A.IDX[f"{s['prefijo']}Thumb{k}"]
        out[f"{s['prefijo']}Thumb{k}"] = A.REL[j] @ Rot.from_rotvec(eje_pulgar*np.radians(a*escala)).as_matrix()
    return out

def _sin_tocar(L, prefijo, A):
    """True si los dedos de esa mano siguen en la pose relajada por defecto (nadie los animó)."""
    js = [A.IDX[n] for n in A.N77 if n.startswith(prefijo) and n != prefijo and not n.endswith('End')]
    return np.abs(L[:, js] - A.REL[js]).max(axis=(1, 2, 3)) < 1e-4

def aplicar(L77, escala=1.0):
    A, xb = _datos()
    L = np.array(L77, float).copy()
    for lado in ('izq', 'der'):
        pref = A.LADOS[lado]['prefijo']; libre = _sin_tocar(L, pref, A)
        if not libre.any(): continue
        for nom, R in pose_mano(lado, escala).items():
            L[libre, A.IDX[nom]] = R
    return L
