"""Max v4 en el motor (archivo NUEVO; no cambia nada de lo que ya hay). Es MaxV3 con un arreglo importante:

ARREGLO: MaxV3 calculaba la postura neutra (hombros relajados, cuello recto) con giros medidos en los BVH que salen
de Kimodo directo. Los BVH que escribe escena.py (ZIP_18/21: exportar_bvh) tienen otro reposo (los huesos en otra
dirección), y con esos BVH MaxV3 deformaba el cuerpo: hombros encogidos hacia atrás, trapecio subido y el cuello
torcido en varios cuadros (MaxSoma con max_v1 sí los mostraba bien). MaxV4 calcula la corrección de postura UNA vez,
en el espacio de Max, y la pone encima del enlace de MaxSoma; como MaxSoma ya se adapta al reposo de cada BVH, la
postura corregida ahora sale igual con cualquier BVH (de Kimodo directo o de escena.py). Medido: con los BVH de Kimodo
da lo mismo que MaxV3 (diferencia 0), y con los de escena.py el cuerpo queda como el de MaxSoma pero con la postura
nueva.

ARREGLO 2 (hombro derecho): los BVH de Kimodo traen la clavícula derecha girada ~155° sobre su propio eje respecto
al reposo (convención de espejo del lado derecho; la izquierda viene a ~5°). Con eso la piel del hombro derecho se
retorcía: pliegue y "aleta" encima del hombro y el trapecio deformado (pasaba también en MaxSoma, tapado por la
sudadera). MaxV4 quita ese giro de convención (ROLL_CLAVICULA_DER). Girar sobre el eje del hueso no mueve ninguna
articulación: brazos, manos y agarres quedan exactamente en el mismo lugar.

Todo lo demás es igual que MaxV3 (camiseta, piel tapada, manos donde las pone agarres_ik, pantalón, física opcional):
    from soma.max_v4 import MaxV4
    max = MaxV4(glb='/home/claude/Personajes/Max/max_v4.glb')                  # con camiseta
    max = MaxV4(glb='/home/claude/Personajes/Max/max_v4.glb', camiseta=False)  # sin camisa
Para la cara viva, usar MaxCara (soma/cara.py), que ya es un MaxV4.
"""
import numpy as np

from .rig import MaxSoma, TPOSE, _M
from .bvh import Clip
from .max_v3 import MaxV3


class MaxV4(MaxV3):
    PIEL_ABERTURA = 0.035     # camiseta ajustada: solo la piel a menos de 3,5 cm del cuello y las mangas se dibuja
                              # debajo de la tela (con 12 cm asomaban puntitos de piel al mover los brazos)

    def _correccion(s):
        """Giro por hueso, en el marco de Max: postura MaxV3 = correccion · enlace de MaxSoma (con el reposo nativo)."""
        if getattr(s, '_corr', None) is None:
            ref = Clip(TPOSE)
            bv3 = MaxV3._bind(s, ref)
            bso = MaxSoma._bind(s, ref)
            s._corr = np.einsum('jab,jcb->jac', bv3['A'], bso['A'])       # A_v3 · A_soma^T
        return s._corr

    ROLL_CLAVICULA_DER = 158.0   # grados (medido: der 152-156°, izq 3-6° -> simétrico con la izquierda)

    def _bind(s, clip):
        clave = (clip.reposo.round(5).tobytes(), tuple(clip.nombres), 'v4', s.ROLL_CLAVICULA_DER, tuple(sorted(s.NEUTRO.items())),
                 tuple(sorted(s.IGUALAR.items())))
        if clave in s._cache:
            return s._cache[clave]
        C = s._correccion()
        bs = MaxSoma._bind(s, clip)
        Al = np.einsum('jab,jbc->jac', C, bs['A'])
        Pl = bs['P']
        if s.ROLL_CLAVICULA_DER:
            j, h = s.idx['RightShoulder'], s.idx['RightArm']
            e = Pl[h] - Pl[j]; e = e / np.linalg.norm(e)
            g = np.radians(s.ROLL_CLAVICULA_DER)
            K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            Q = np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K
            Al = Al.copy(); Al[j] = Q @ Al[j]
        off = np.zeros_like(Pl)
        for j, p in enumerate(s.padres):
            if p >= 0:
                off[j] = Al[p].T @ (Pl[j] - Pl[p])
        ibm = np.stack([np.linalg.inv(_M(Al[j], Pl[j])) for j in range(len(Al))])
        b = dict(A=Al, P=Pl, off=off, ibm=ibm, k=bs['k'], hips0=bs['hips0'])
        s._cache[clave] = b
        return b
