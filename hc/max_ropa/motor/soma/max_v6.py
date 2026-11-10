"""Max v6 (ZIP_28, archivo NUEVO: no cambia nada de lo que ya hay). Es MaxV4 / MaxCara con el BRAZO DERECHO arreglado
y los HOMBROS PAREJOS. Reemplaza a MaxV5 del ZIP 27 (que puede quedarse instalado; no choca).

QUÉ ARREGLA: los BVH de Kimodo traen toda la cadena del brazo derecho (clavícula, brazo y antebrazo) girada ~172°
sobre el eje de cada hueso respecto al reposo (convención de espejo del lado derecho; el izquierdo viene casi en 0).
La mano derecha sí viene bien, así que todo ese giro se juntaba en la muñeca y en el hombro: brazo aplastado,
muñeca estrangulada, manga retorcida y "aleta" encima del hombro. (Pasaba también con MaxSoma y max_v1, tapado por
la sudadera.) El ZIP 26 solo corregía la clavícula; MaxV5 (ZIP 27) corregía la cadena completa.

HOMBROS PAREJOS (nuevo en MaxV6): la corrección de postura de MaxV4 baja las clavículas para relajar los hombros.
En el lado derecho se aplicaba DESPUÉS del giro de convención de Kimodo, así que quedaba al revés: el hombro derecho
SUBÍA ~6 cm en vez de bajar (de pie: izq. 1,44 m / der. 1,50 m). MaxV6 aplica el giro de convención antes de la
corrección: de pie quedan parejos (1,436 / 1,441 m). Además, con el hombro en su sitio, la mano derecha ya alcanza
los agarres: mano a menos de 5 mm de donde la pone agarres_ik en toda la escena de ejemplo (antes hasta 31 mm).
Si en una animación un brazo sube más que el otro, es porque Kimodo lo hizo así (ej.: en celebrar, Kimodo sube el
hombro derecho 5,6 cm más que el izquierdo); Max lo copia igual.
Medido en 2 clips: giro de la mano der. respecto al antebrazo, de -171° a 0°; clavícula der. respecto al pecho, de
172° a 0° (el lado izquierdo da entre -13° y 8°).

Las manos siguen yendo a donde las pone agarres_ik (IK del brazo de MaxV3).

Uso (igual que MaxCara / MaxV4, solo cambia el nombre):
    from soma.max_v6 import MaxCaraV6, MaxV6
    from soma.cara import PistaCara
    m = MaxCaraV6(glb='/home/claude/Personajes/Max/max_v4.glb')          # cara viva + brazo arreglado
    m.pista = PistaCara.desde_escena('<carpeta salida de escena.py>', '<escena.json>')
    m = MaxV6(glb='/home/claude/Personajes/Max/max_v4.glb')              # solo cuerpo, sin cara viva
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
        clave = ('v6', id(b), tuple(sorted(s.ROLL.items())))
        c = s.__dict__.setdefault('_cache_v6', {})
        if clave in c:
            return c[clave][1]
        A = b['A'].copy(); P = b['P']; ix = s.idx
        C = s._correccion()               # corrección de postura de MaxV4 (en el espacio de Max)
        hijo = {'RightShoulder': 'RightArm', 'RightArm': 'RightForeArm', 'RightForeArm': 'RightHand'}
        for n, gr in s.ROLL.items():
            if not gr:
                continue
            j, h = ix[n], ix[hijo[n]]
            # el giro de convención va ANTES de la corrección de postura: eje = C·(eje del hueso en reposo).
            # (Si se pone después, la corrección "bajar la clavícula" queda girada 172° y el hombro derecho SUBE.)
            e = C[j] @ (P[h] - P[j]); e = e / np.linalg.norm(e)
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


class MaxV6(_BrazoDerecho, MaxV4):
    """MaxV4 con el brazo derecho arreglado y los hombros parejos."""


class MaxCaraV6(_BrazoDerecho, MaxCara):
    """MaxCara (cara viva, cuello, ojos) con el brazo derecho arreglado y los hombros parejos."""
