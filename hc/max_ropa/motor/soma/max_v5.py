"""Max v5 (ZIP_27, archivo NUEVO: no cambia nada de lo que ya hay). Es MaxV4 / MaxCara con el BRAZO DERECHO arreglado.

QUÉ ARREGLA: los BVH de Kimodo traen toda la cadena del brazo derecho (clavícula, brazo y antebrazo) girada ~172°
sobre el eje de cada hueso respecto al reposo (convención de espejo del lado derecho; el izquierdo viene casi en 0).
La mano derecha sí viene bien, así que todo ese giro se juntaba en la muñeca y en el hombro: brazo aplastado,
muñeca estrangulada, manga retorcida y "aleta" encima del hombro. (Pasaba también con MaxSoma y max_v1, tapado por
la sudadera.) El ZIP 26 solo corregía la clavícula; MaxV5 corrige la cadena completa.
Medido en 2 clips: giro de la mano der. respecto al antebrazo, de -171° a 0°; clavícula der. respecto al pecho, de
172° a 0° (el lado izquierdo da entre -13° y 8°).

Girar un hueso sobre su propio eje NO mueve ninguna articulación: hombros a la misma altura y brazos, manos y
agarres exactamente en el mismo lugar (medido: 0,000 mm de diferencia con MaxV4).

Uso (igual que MaxCara / MaxV4, solo cambia el nombre):
    from soma.max_v5 import MaxCaraV5, MaxV5
    from soma.cara import PistaCara
    m = MaxCaraV5(glb='/home/claude/Personajes/Max/max_v4.glb')          # cara viva + brazo arreglado
    m.pista = PistaCara.desde_escena('<carpeta salida de escena.py>', '<escena.json>')
    m = MaxV5(glb='/home/claude/Personajes/Max/max_v4.glb')              # solo cuerpo, sin cara viva
"""
import numpy as np

from .max_v4 import MaxV4
from .cara import MaxCara


class _BrazoDerecho:
    # grados de giro de convención, sobre el eje propio de cada hueso
    ROLL = {'RightShoulder': 172.0, 'RightArm': 172.0, 'RightForeArm': 172.0}
    ROLL_CLAVICULA_DER = 0.0          # el arreglo parcial del ZIP 26 queda reemplazado por ROLL

    def _bind(s, clip):
        b = super()._bind(clip)
        clave = ('v5', id(b), tuple(sorted(s.ROLL.items())))
        c = s.__dict__.setdefault('_cache_v5', {})
        if clave in c:
            return c[clave][1]
        A = b['A'].copy(); P = b['P']; ix = s.idx
        hijo = {'RightShoulder': 'RightArm', 'RightArm': 'RightForeArm', 'RightForeArm': 'RightHand'}
        for n, gr in s.ROLL.items():
            if not gr:
                continue
            j, h = ix[n], ix[hijo[n]]
            e = P[h] - P[j]; e = e / np.linalg.norm(e)
            g = np.radians(gr)
            K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            A[j] = (np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K) @ A[j]
        off = np.zeros_like(P)
        for j, p in enumerate(s.padres):
            if p >= 0:
                off[j] = A[p].T @ (P[j] - P[p])
        ibm = b['ibm'].copy()
        for n in s.ROLL:
            j = ix[n]
            M = np.eye(4); M[:3, :3] = A[j]; M[:3, 3] = P[j]
            ibm[j] = np.linalg.inv(M)
        nb = dict(b); nb.update(A=A, off=off, ibm=ibm)
        c[clave] = (b, nb)            # se guarda b para que id(b) no se reutilice
        return nb


class MaxV5(_BrazoDerecho, MaxV4):
    """MaxV4 con el brazo derecho arreglado."""


class MaxCaraV5(_BrazoDerecho, MaxCara):
    """MaxCara (cara viva, cuello, ojos) con el brazo derecho arreglado."""
