"""La esposa (Rocketbox F08) con su taza: la taza va en la mano izquierda (entre las dos manos cuando la sostiene con las
dos), los dedos la rodean, y puede saludar con la derecha sin levantar la vista."""
import math, numpy as np
from tp_base import *
import actor_rb
import recursos_vel as RV

TAZA_ALTO = .10


class Esposa:
    def __init__(s, E, clave='esposa'):
        s.E = E; s.A = actor_rb.ActorRB(E.r, 'rb_mujer_f08', clave, 1.64)
        s.c_taza = Clip(ruta_anim('tp_esposa_taza_quieta_1')); s.c_saludo = Clip(ruta_anim('g18_saludar'))
        s._agarre()

    # -------------------------------------------------------------------------------------------- la taza en la mano
    def _taza_modelo(s):
        """Radio del cuerpo de la taza, dónde está su eje y hacia dónde apunta el asa (en el modelo, ya a escala)."""
        m = RV._partes('tp_taza'); P = np.concatenate([p['P'] for p in m['partes']]); esc = TAZA_ALTO / P[:, 2].max()
        cuerpo = (P[:, 2] > P[:, 2].max() * .25) & (P[:, 2] < P[:, 2].max() * .75) & (np.abs(P[:, 0]) > P[:, 0].max() * .9)
        cy = float(P[cuerpo, 1].mean()); r = float(P[:, 0].max())
        asa = 1.0 if abs(P[:, 1].max() - cy) > abs(P[:, 1].min() - cy) else -1.0
        return r * esc, np.array([0, cy * esc, 0]), asa

    def _agarre(s):
        """La mano izquierda rodea el cuerpo de la taza: el eje de la taza queda a (radio + 1 cm) de la palma, del lado de
        la palma, y los dedos se cierran justo hasta tocarla (medido: la punta del dedo medio a 6 mm de la cerámica)."""
        A = s.A; s.r_taza, s.off_taza, s.asa = s._taza_modelo()
        A.objetos(s.c_taza, 1.45, (0, 0, 0), 0.0, dedos={'L': 0})
        Wl = A.hueso_mat('Bip01 L Hand'); h = Wl[:3, 3]
        f2 = A.hueso('Bip01 L Finger2'); f1 = A.hueso('Bip01 L Finger1'); f4 = A.hueso('Bip01 L Finger4')
        x = f2 - h; x /= np.linalg.norm(x); y = f1 - f4; y -= x * (y @ x); y /= np.linalg.norm(y)
        n = np.cross(x, y); n[2] = 0
        if n @ np.array([-np.sign(h[0]), 0, 0]) < 0: n = -n                       # la palma mira hacia el centro del cuerpo
        n /= np.linalg.norm(n)
        palma = h + (f2 - h) * .55
        c = palma + n * (s.r_taza + .010); c[2] = palma[2]
        s.local = Wl[:3, :3].T @ (c - h); s.n_local = Wl[:3, :3].T @ n
        s.cierre = {0: 40}; s.dist_dedos = {}                                       # cada dedo se cierra justo hasta tocarla
        for k in (1, 2, 3, 4):
            mejor = None
            for ang in range(0, 112, 3):
                A.objetos(s.c_taza, 1.45, (0, 0, 0), 0.0, dedos={'L': {k: ang}})
                W = A.hueso_mat('Bip01 L Hand'); cc = W[:3, 3] + W[:3, :3] @ s.local
                a1, a2 = A.hueso(f'Bip01 L Finger{k}1'), A.hueso(f'Bip01 L Finger{k}2'); tip = a2 + (a2 - a1) * .85
                dd = [np.linalg.norm((p - cc)[:2]) - s.r_taza for p in (a1, a2, tip)]
                err = abs(dd[2] - .006) + 4 * max(0.0, .002 - min(dd))              # la punta a 6 mm, ninguna falange adentro
                if mejor is None or err < mejor[0]: mejor = (err, ang, dd[2])
            s.cierre[k] = mejor[1]; s.dist_dedos[k] = mejor[2]
        s.cierre[0] = float(np.mean([s.cierre[k] for k in (1, 2, 3, 4)]))

    def objetos(s, t, pos, giro, saluda=None, cara='neutral', mirar=None):
        """saluda = (t0, t1): saluda con la derecha entre esos tiempos (entra y sale suave).
        mirar = punto del mundo: levanta la cabeza y la gira hacia ahí (los ojos también)."""
        tc = 1.45 + .10 * math.sin(t * 2.0)
        capa = None; cierre_d = 38; w = 0.0
        if saluda is not None:
            w = suave((t - saluda[0]) / .45) * (1 - suave((t - saluda[1] + .42) / .42))
            if w > 0:
                capa = ('saludo', (t - saluda[0]) * 2 * math.pi / .95, 'Right', w); cierre_d = 38 * (1 - w) + 6 * w
        mira = None
        if mirar is not None:
            wm = 1.0 if saluda is None else suave((t - saluda[0] + .30) / .45) * (1 - suave((t - saluda[1] + .15) / .45))
            if wm > 0:
                hd = getattr(s, '_cabeza', None)
                if hd is None: s.A.objetos(s.c_taza, tc, tuple(pos), giro); hd = s._cabeza = s.A.hueso('Bip01 Head')
                d = np.asarray(mirar, float) - hd; a = math.radians(giro)
                Rg = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.]])
                dl = Rg.T @ d; mira = (np.array([dl[0], dl[2], -dl[1]]), wm)                 # mundo -> marco del BVH (Y arriba, Z adelante)
        if mira is not None and isinstance(cara, str):
            cara = dict(actor_rb.EXPR[cara]); cara.update({'AK_11_EyeLookDownLeft': 0.0, 'AK_12_EyeLookDownRight': 0.0, 'AK_44_MouthSmileLeft': .55 * mira[1], 'AK_45_MouthSmileRight': .55 * mira[1]})
        o = s.A.objetos(s.c_taza, tc, tuple(pos), giro, cara=cara, capa=capa, dedos={'L': s.cierre, 'R': cierre_d}, mirar=mira)
        W = s.A.hueso_mat('Bip01 L Hand'); c = W[:3, 3] + W[:3, :3] @ s.local
        nw = W[:3, :3] @ s.n_local; nw[2] = 0; nw /= np.linalg.norm(nw)
        g = math.degrees(math.atan2(nw[1], nw[0])) - 90 * s.asa                  # el asa hacia el lado libre (lejos de la palma)
        R = np.array([[math.cos(math.radians(g)), -math.sin(math.radians(g)), 0], [math.sin(math.radians(g)), math.cos(math.radians(g)), 0], [0, 0, 1.]])
        base = c - R @ s.off_taza - np.array([0, 0, TAZA_ALTO * .45])
        return o + RV.modelo(s.E.r, 'tp_taza', tuple(base), giro_z=g, alto=TAZA_ALTO)
