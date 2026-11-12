"""T1 · Gancho estilo Goku (tiempo viejo 0,0–6,3 s).
0,00 cocina: Max se pone el índice y el medio en la frente y se concentra ("Si pudieras teletransportarte")
1,45 parpadea y desaparece con un destello (efecto de teletransportación instantánea)
1,65 aparece en una calle de Tokio de noche, feliz, mirando alrededor ("a cualquier parte del mundo")
3,45 se desploma al piso ("no llegarías vivo")
4,70 todo se rebobina hasta la cocina ("ni siquiera llegarías")
5,75 vuelve a estar en la cocina, en el mismo lugar, y arranca a caminar a la cabina (sigue en la T2)."""
import math, os, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, tokio as TK, ik_tp as IK

GOKU = os.environ.get('GOKU', 'tp_goku_frente_2')
MIRAR = os.environ.get('MIRAR', 'tp_mirar_alrededor_feliz_1')
CAER = 'mq_mq37_desmayo_a'
T_VA, T_LLEGA, T_CAE, T_REB, T_VUELVE = 1.45, 1.65, 3.45, 4.70, 5.75
P_TK = np.array([-3.2, .25, 0.0]); YAW_TK = 15.0                  # en medio de la calle de Tokio, mirando a la cámara


def _clip_o(E, n, alt):
    try: clip(E, n); return n
    except FileNotFoundError: return alt


class Toma:
    INI, FIN = 0.0, 6.3

    def preparar(s, E):
        s.E = E
        C.preparar(E); s.fijos_casa = K.fijos(E)
        s.ini = K.inicio(E); s.ini[2] = C.Z0
        s.goku = _clip_o(E, GOKU, 'g16_espera_respirando'); s.mirar = _clip_o(E, MIRAR, 'g18_saludar')
        s.p_goku = colocar(E, s.goku, 0.0, s.ini, K.CAM_YAW)
        s.p_quieto = colocar(E, 'g16_espera_respirando', 1.0, s.ini, K.CAM_YAW)
        s.pista = caras([(-9, 1.65, 'decidido'), (1.65, 3.45, 'euforia'), (3.45, 3.9, 'sorpresa'), (3.9, 99, 'neutral')])
        s.tk = None

    def _tokio(s, E):
        if s.tk is None:
            guard = (E.cielo, getattr(E, 'zsuelo', 0.0), getattr(E, 'calido', 0.0), dict(getattr(E, 'grado', {}) or {}))
            TK.preparar(E); s.tk = (TK.escena(E, 0.0, cabina=False), (E.cielo, E.zsuelo, E.calido, dict(E.grado or {})))
            E.cielo, E.zsuelo, E.calido, E.grado = guard
        return s.tk

    # ---------------------------------------------------------------- Max en cada parte
    def _max_casa(s, t):
        E = s.E
        if t < T_VA + .2:                                          # dedos a la frente
            E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0, 'browDownLeft': .6, 'browDownRight': .6} if t > .62 else {}
            w = suave((t - .12) / .50)                                  # índice y medio a la frente (IK), y ahí se queda
            return max_objs(E, s.goku, (.35 + t * .95) if t < .7 else (1.015 + (t - .7) * .2), s.p_goku, K.CAM_YAW, pista=s.pista, t_escena=t,
                            suelo=C.Z0, ik=IK.goku(w))
        E.max.cara_manual = {}
        return K.caminar(E, K.T_CAMINA0, s.pista)                 # vuelve ya en el primer paso de la caminata (la T2 sigue sin salto)

    def _max_tokio(s, t):
        E = s.E; E.max.cara_manual = {}
        if t < T_CAE:
            n = s.mirar; tc = .2 + (t - T_LLEGA) * 1.0
            return max_objs(E, n, min(tc, clip(E, n).dur - .05), P_TK, YAW_TK, pista=s.pista, t_escena=t, suelo=0.0)
        if t > 3.95: E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0, 'jawOpen': .25}
        tc = .6 + (t - T_CAE) * 1.6                                     # se desploma (más rápido que el clip original)
        return max_objs(E, CAER, min(tc, clip(E, CAER).dur - .05), P_TK, YAW_TK, pista=s.pista, t_escena=t, suelo=0.0)

    # ---------------------------------------------------------------- efecto de teletransportación
    def _destello(s, centro, k):
        """Línea de luz vertical donde estaba (o adonde llega) Max: se afina y se apaga."""
        if k <= 0: return []
        M = np.eye(4, dtype=np.float32); M[:3, :3] = np.diag([.018 * k + .004, .018 * k + .004, 2.0 * (.4 + .6 * k)]); M[:3, 3] = centro + (0, 0, 1.0)
        Mg = M.copy(); Mg[:3, :3] = np.diag([.10 * k, .10 * k, 2.1 * (.4 + .6 * k)])
        return [dict(mesh='box', model=M, texture='white', color=(.85, .95, 1.0), emission=12 * k, noshadow=True),
                dict(mesh='box', model=Mg, texture='white', color=(.6, .85, 1.0, .14 * k), emission=2.5 * k, transparent=True, noshadow=True)]

    def _parpadeo(s, t, t0, t1, aparece):
        """Max se ve a ratos (como en la tele vieja): True = dibujarlo."""
        if t < t0: return not aparece
        if t >= t1: return aparece
        k = int((t - t0) * 60) // 2
        return (k % 2 == 0) if not aparece else (k % 2 == 1)

    # ---------------------------------------------------------------- cuadro en tiempo "de la escena" (el rebobinado lo reusa)
    def _escena(s, E, t):
        if t < T_LLEGA or t >= T_VUELVE:                                 # en la cocina
            o, M, J = s._max_casa(t)
            h = J['Head'] if t < T_LLEGA else s.ini + (0, 0, 1.65)
            ver = s._parpadeo(t, T_VA, T_VA + .2, aparece=False) if t < T_VUELVE else s._parpadeo(t, T_VUELVE, T_VUELVE + .15, aparece=True)
            objs = s.fijos_casa + (o if ver else [])
            if T_VA <= t < T_VA + .35: objs += s._destello(s.ini, 1 - (t - T_VA) / .35)
            if T_VUELVE <= t < T_VUELVE + .3: objs += s._destello(s.ini, 1 - (t - T_VUELVE) / .3)
            fr = -K.F; lado = -K.S
            if t < T_LLEGA:
                u = suave(t / T_LLEGA); hh = s.ini + (0, 0, 1.62)
                cam = (tuple(hh + fr * (1.45 - .25 * u) + lado * .65 + (0, 0, -.02)), tuple(hh + (0, 0, -.02)), 42)
            else:                                                       # vuelve: el mismo plano de la T2 (travelling lateral)
                hp = J['Hips']                                         # exactamente la cámara con la que empieza la T2
                cam = (tuple(hp + -K.S * 2.4 + K.F * .2 + (0, 0, .25) + K.F * .3), tuple(hp + (0, 0, .2) - K.F * .25), 46)
            E.posts = []; E.grado = {}
            return E.render(objs, cam, C.luz(cam[1], cam[0]))
        # en Tokio
        fijos, conf = s._tokio(E)
        guard = (E.cielo, getattr(E, 'zsuelo', 0.0), getattr(E, 'calido', 0.0), dict(getattr(E, 'grado', {}) or {}))
        E.cielo, E.zsuelo, E.calido, E.grado = conf[0], conf[1], conf[2], dict(conf[3])
        o, M, J = s._max_tokio(t)
        ver = s._parpadeo(t, T_LLEGA, T_LLEGA + .2, aparece=True)
        objs = fijos + (o if ver else [])
        if T_LLEGA <= t < T_LLEGA + .35: objs += s._destello(P_TK, 1 - (t - T_LLEGA) / .35)
        F = np.array([math.sin(math.radians(YAW_TK)), -math.cos(math.radians(YAW_TK)), 0])
        hp = J['Hips']
        if t < T_CAE:
            cam = (tuple(P_TK + F * 2.6 + np.array([.6, 0, 1.45])), tuple(P_TK + (0, 0, 1.15)), 46)
        else:                                                           # baja con él al piso
            v = suave((t - T_CAE) / (T_REB - T_CAE))
            cam = (tuple(P_TK + F * (2.6 - .5 * v) + np.array([.6 + .3 * v, 0, 1.45 - .9 * v])), tuple(mezcla(P_TK + (0, 0, 1.15), hp + (0, 0, -.1), v)), 46)
        E.grado = dict(E.grado or {}, foco=tuple(hp), cm=cam, k_dof=1.2)
        E.posts = []
        Lz = TK.luz(cam[1], cam[0]); Lz['env_k'] = .6                 # el gancho tiene que leerse en un cuadro: Max bien iluminado
        Lz['lights'] += [dict(pos=tuple(P_TK + F * 2.0 + np.array([-.8, 0, 2.3])), col=(3.0, 2.7, 2.3)), dict(pos=tuple(P_TK - F * 1.5 + np.array([1.2, 0, 2.6])), col=(1.2, 1.6, 2.4))]
        img = E.render(objs, cam, Lz)
        E.cielo, E.zsuelo, E.calido, E.grado = guard
        return img

    def cuadro(s, E, t):
        if T_REB <= t < T_VUELVE:                                       # rebobinado: la escena hacia atrás hasta la cocina
            u = (t - T_REB) / (T_VUELVE - T_REB)
            tt = T_REB - (T_REB - (T_VA - .05)) * suave(u)              # de Tokio de vuelta al momento en que se fue
            img = s._escena(E, tt)
            return rebobinado(img, min(1.0, .5 + u * 1.5) if u < .85 else (1 - u) / .15, t)
        img = s._escena(E, t)
        if T_VA <= t < T_VA + .2 or T_LLEGA <= t < T_LLEGA + .2 or T_VUELVE <= t < T_VUELVE + .15:
            img = onda(img, t)
        return img


def onda(img, t):
    """Distorsión cortita del aire (la imagen 'tiembla' al teletransportarse)."""
    H, W = img.shape[:2]; y = np.arange(H)
    dx = (np.sin(y * .08 + t * 90) * 7 * (W / 1080)).astype(int)
    out = img[y[:, None], (np.arange(W)[None, :] + dx[:, None]) % W]
    a = out.astype(np.float32); a[..., 2] = np.roll(a[..., 2], int(4 * W / 1080), axis=1); a[..., 0] = np.roll(a[..., 0], -int(4 * W / 1080), axis=1)
    return np.clip(a * 1.08, 0, 255).astype(np.uint8)
