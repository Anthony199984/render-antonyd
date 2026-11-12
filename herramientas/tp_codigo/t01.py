"""T1 · Gancho estilo Goku + caja de muertos (tiempo viejo 0,0–6,3 s).
0,00 cocina: Max se pone el índice y el medio en la frente y se concentra ("Si pudieras teletransportarte")
1,45 parpadea y desaparece con un destello (efecto de teletransportación instantánea)
1,65 aparece en una calle de Tokio de noche, feliz, mirando alrededor ("a cualquier parte del mundo")
3,30 corte: las dos mitades de una caja de muertos llegan volando desde esquinas opuestas ("no llegarías vivo")
3,72 se cierran de golpe con Max adentro, aterrado; la cámara se mete: la tapa se desvanece y queda su cara en el forro
4,45 cierra los ojos; negro ("ni siquiera llegarías")
4,76 abre los ojos en la cocina, en el mismo lugar del principio: desorientado, mira a todos lados
6,30 corte a la T2: camina hacia la cabina ("porque entras en la cabina de tu casa")."""
import math, os, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, tokio as TK, ik_tp as IK, ataud as AT

GOKU = os.environ.get('GOKU', 'tp_goku_frente_2')
MIRAR = os.environ.get('MIRAR', 'tp_mirar_alrededor_feliz_1')
MIEDO = 'tp_tapar_boca_1'                                        # dentro de la caja: las manos suben a la boca (cabe sin atravesarla: medido)
T_VA, T_LLEGA = 1.45, 1.65
MIRA0 = float(os.environ.get('MIRA0', 0.0))                         # desde qué momento del clip despierta (la cabeza de frente)
T_CAJA, T_CIERRA, T_TAPA0, T_TAPA1, T_OJOS, T_NEGRO, T_CASA = 3.30, 3.72, 3.92, 4.22, 4.42, 4.60, 4.76
P_TK = np.array([-3.2, .25, 0.0]); YAW_TK = 15.0                  # en medio de la calle de Tokio, mirando a la cámara


def _clip_o(E, n, alt):
    try: clip(E, n); return n
    except FileNotFoundError: return alt


class Toma:
    INI, FIN = 0.0, 6.3

    def preparar(s, E):
        s.E = E
        C.preparar(E); s.fijos_casa = K.fijos(E); AT.preparar(E.r)
        s.ini = K.inicio(E); s.ini[2] = C.Z0
        s.goku = _clip_o(E, GOKU, 'g16_espera_respirando'); s.mirar = _clip_o(E, MIRAR, 'g18_saludar')
        s.p_goku = colocar(E, s.goku, 0.0, s.ini, K.CAM_YAW)
        s.p_quieto = colocar(E, 'g16_espera_respirando', 1.0, s.ini, K.CAM_YAW)
        s.pista = caras([(-9, 1.65, 'decidido'), (1.65, 3.30, 'euforia'), (3.30, 4.70, 'susto'), (4.70, 99, 'confundido')])
        s.tk = None
        s.p_mira = colocar(E, s.mirar, MIRA0, s.ini, K.CAM_YAW)          # despierta en el mismo sitio donde empezó
        s._ojos_casa = max_objs(E, s.mirar, MIRA0, s.p_mira, K.CAM_YAW, suelo=C.Z0)[2]['Head'] + (0, 0, .075)
        s.caja = None

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
            E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0, 'browDownLeft': .6, 'browDownRight': .6} if t > .50 else {}
            w = suave((t + .02) / .42)                                  # índice y medio a la frente (IK), y ahí se queda
            tg = t + .12                                                # el video arranca con la mano ya subiendo (gancho desde el cuadro 0)
            return max_objs(E, s.goku, (.35 + tg * .95) if tg < .7 else (1.015 + (tg - .7) * .2), s.p_goku, K.CAM_YAW, pista=s.pista, t_escena=t,
                            suelo=C.Z0, ik=IK.goku(w))
        # despierta en la cocina: abre los ojos de golpe y mira a un lado y al otro, perdido
        k = 1 - suave((t - T_CASA - .04) / .22)
        E.max.cara_manual = {'eyeBlinkLeft': k, 'eyeBlinkRight': k, 'browInnerUp': .55 * (1 - k)}
        n = s.mirar; tc = MIRA0 + (t - T_CASA) * 1.2                  # cabeza de frente, mira a un lado y después al otro
        o = max_objs(E, n, min(tc, clip(E, n).dur - .05), s.p_mira, K.CAM_YAW, pista=s.pista, t_escena=t, suelo=C.Z0)
        E.max.cara_manual = {}
        return o

    def _max_tokio(s, t):
        E = s.E; E.max.cara_manual = {}
        if t < T_CAJA:
            n = s.mirar; tc = .2 + (t - T_LLEGA) * 1.0
            return max_objs(E, n, min(tc, clip(E, n).dur - .05), P_TK, YAW_TK, pista=s.pista, t_escena=t, suelo=0.0)
        if t > T_OJOS: k = suave((t - T_OJOS) / .16); E.max.cara_manual = {'eyeBlinkLeft': k, 'eyeBlinkRight': k}
        o = max_objs(E, MIEDO, .64 + (t - T_CAJA) * .26, P_TK, YAW_TK, pista=s.pista, t_escena=t, suelo=0.0)
        E.max.cara_manual = {}
        return o

    def _centro_caja(s):
        """Dónde se cierra la caja: debajo de Max, centrada en el fondo de su cuerpo (medido con la malla)."""
        if s.caja is None:
            E = s.E; o, M, J = max_objs(E, MIEDO, .9, P_TK, YAW_TK, suelo=0.0)
            Pl = np.concatenate([p for p, _ in E.max._ult_pose[2]]); Pw = Pl @ M[:3, :3].T + M[:3, 3]
            c0 = np.array([J['Hips'][0], J['Hips'][1], 0.0]); L = (Pw - c0) @ AT._R(YAW_TK)
            s.caja = c0 + AT._R(YAW_TK) @ np.array([(L[:, 0].min() + L[:, 0].max()) / 2, (L[:, 1].min() + L[:, 1].max()) / 2, 0])
        return s.caja

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

    # ---------------------------------------------------------------- cuadros
    def _casa(s, E, t):
        o, M, J = s._max_casa(t)
        ver = s._parpadeo(t, T_VA, T_VA + .2, aparece=False) if t < T_LLEGA else True
        objs = s.fijos_casa + (o if ver else [])
        if T_VA <= t < T_VA + .35: objs += s._destello(s.ini, 1 - (t - T_VA) / .35)
        fr = -K.F; lado = -K.S
        if t < T_LLEGA:
            u = suave(t / T_LLEGA); hh = s.ini + (0, 0, 1.62)
            cam = (tuple(hh + fr * (1.45 - .25 * u) + lado * .65 + (0, 0, -.02)), tuple(hh + (0, 0, -.02)), 42)
        else:                                                           # abre los ojos: de su cara se abre al plano medio en la cocina
            u = suave((t - T_CASA - .15) / (s.FIN - T_CASA - .15)); h = J['Head']
            a = math.radians(K.CAM_YAW); Fm = np.array([math.sin(a), -math.cos(a), 0]); Sm = np.array([-Fm[1], Fm[0], 0])
            ojos = s._ojos_casa                                         # el mismo encuadre que en la caja (corte a la misma cara)
            ojo = ojos + Fm * (.58 + 1.10 * u) + Sm * (.03 + .40 * u) + (0, 0, .02 - .12 * u)
            cam = (tuple(ojo), tuple(mezcla(ojos, h + (0, 0, -.30), u)), 36 + 10 * u)
        E.posts = []; E.grado = {}
        return E.render(objs, cam, C.luz(cam[1], cam[0]))

    def _tokio_cuadro(s, E, t):
        fijos, conf = s._tokio(E)
        guard = (E.cielo, getattr(E, 'zsuelo', 0.0), getattr(E, 'calido', 0.0), dict(getattr(E, 'grado', {}) or {}))
        E.cielo, E.zsuelo, E.calido, E.grado = conf[0], conf[1], conf[2], dict(conf[3])
        o, M, J = s._max_tokio(t)
        ver = s._parpadeo(t, T_LLEGA, T_LLEGA + .2, aparece=True)
        objs = fijos + (o if ver else [])
        if T_LLEGA <= t < T_LLEGA + .35: objs += s._destello(P_TK, 1 - (t - T_LLEGA) / .35)
        F = np.array([math.sin(math.radians(YAW_TK)), -math.cos(math.radians(YAW_TK)), 0]); Sd = np.array([-F[1], F[0], 0])
        hp = J['Hips']; Lz = TK.luz((0, 0, 0)); extra = []; tapa = 1.0
        if t < T_CAJA:
            u = suave((t - T_LLEGA) / (T_CAJA - T_LLEGA))                # se acerca despacio y le da la vuelta un poco mientras mira feliz
            cam = (tuple(P_TK + F * (3.0 - .55 * u) + Sd * (.55 - .65 * u) + np.array([0, 0, 1.45 - .12 * u])), tuple(P_TK + (0, 0, 1.15 + .12 * u)), 46)
            foco = hp
        else:
            c = s._centro_caja()
            u = (t - T_CAJA) / (T_CIERRA - T_CAJA)
            golpe = max(0.0, 1 - (t - T_CIERRA) / .30) if t >= T_CIERRA else 0.0
            tapa = 1.0 if t < T_TAPA0 else max(0.0, 1 - (t - T_TAPA0) / (T_TAPA1 - T_TAPA0))
            caja = AT.objetos(u, c, YAW_TK, sacudon=golpe * math.sin((t - T_CIERRA) * 70))
            sin_tapa = objs + [d for d in caja if 'frente' not in d['mesh']]   # la cámara "atraviesa" la tapa: fundido entre con y sin ella
            objs = objs + caja if tapa > 0 else sin_tapa
            h = J['Head']; cara = h + (0, 0, .075)                             # los ojos (se tienen que ver cerrarse)
            ancho = c + F * (3.75 - .40 * suave(min(1.0, u))) + Sd * float(os.environ.get('LADO1', 1.2)) + (0, 0, 1.30)     # todo el ataúd en cuadro; empuja mientras se cierra
            if t < T_CIERRA:
                cam = (tuple(ancho), tuple(c + (0, 0, 1.02)), 48)
            else:
                v = suave((t - T_CIERRA) / (T_OJOS + .05 - T_CIERRA))
                cerca = cara + F * .58 + Sd * .03 + (0, 0, .02)
                ojo = mezcla(ancho, cerca, v); mira = mezcla(c + (0, 0, 1.02), cara, v)
                if golpe > 0: ojo = ojo + np.array([math.sin(t * 91), math.cos(t * 77), math.sin(t * 63) * .5]) * .035 * golpe
                cam = (tuple(ojo), tuple(mira), 48 - 12 * v)
            foco = h if t >= T_CIERRA else c + (0, 0, 1.0)
            if t >= T_TAPA0:                                             # adentro: oscuro, una luz fría de frente y de arriba sobre la cara
                w = 1 - tapa
                extra = [dict(pos=tuple(cara + F * .30 + (0, 0, .28)), col=(1.5 * w, 1.6 * w, 1.9 * w))]
        E.grado = dict(E.grado or {}, foco=tuple(foco), cm=cam, k_dof=1.9 if t < T_CAJA else 2.2)     # con la caja: el letrero del fondo, borroso
        E.posts = []
        Lz = TK.luz(cam[1], cam[0]); Lz['env_k'] = .6                 # el gancho tiene que leerse en un cuadro: Max bien iluminado
        Lz['lights'] += [dict(pos=tuple(P_TK + F * 2.0 + np.array([-.8, 0, 2.3])), col=(3.0, 2.7, 2.3)), dict(pos=tuple(P_TK - F * 1.5 + np.array([1.2, 0, 2.6])), col=(1.2, 1.6, 2.4))]
        if t >= T_TAPA0:
            w = 1 - (1.0 if t < T_TAPA0 else max(0.0, 1 - (t - T_TAPA0) / (T_TAPA1 - T_TAPA0)))
            Lz['env_k'] = .6 - .45 * w; Lz['lights'] = [dict(l, col=tuple(np.array(l['col']) * (1 - .75 * w))) for l in Lz['lights']] + extra
        k = .75 * rampa(t, 3.50, .10, .32, .14)                         # la caja se resalta mientras llega y se cierra (sin aclarar: ya es clara)
        img = render_resaltado(E, objs, cam, Lz, lambda d: str(d.get('mesh', '')).startswith('ataud_'), k, DORADO, crema=0.0)
        if t >= T_CAJA and 0 < tapa < 1:
            b = E.render(sin_tapa, cam, Lz); a_ = suave(tapa)
            img = (img.astype(np.float32) * a_ + b.astype(np.float32) * (1 - a_)).astype(np.uint8)
        E.cielo, E.zsuelo, E.calido, E.grado = guard
        return img

    def cuadro(s, E, t):
        if T_NEGRO <= t < T_CASA:
            return np.zeros((E.r.h, E.r.w, 3), np.uint8)
        if t < T_LLEGA or t >= T_CASA:
            img = s._casa(E, t)
        else:
            img = s._tokio_cuadro(E, t)
            if t > T_OJOS + .08:                                        # cierra los ojos y todo se va a negro
                k = min(1.0, (t - T_OJOS - .08) / (T_NEGRO - T_OJOS - .08))
                img = (img.astype(np.float32) * (1 - k)).astype(np.uint8)
        if t >= T_CASA and t < T_CASA + .22:                            # abre los ojos: sale del negro
            k = (t - T_CASA) / .22; img = (img.astype(np.float32) * k).astype(np.uint8)
        if T_VA <= t < T_VA + .2 or T_LLEGA <= t < T_LLEGA + .2:
            img = onda(img, t)
        return img


def onda(img, t):
    """Distorsión cortita del aire (la imagen 'tiembla' al teletransportarse)."""
    H, W = img.shape[:2]; y = np.arange(H)
    dx = (np.sin(y * .08 + t * 90) * 7 * (W / 1080)).astype(int)
    out = img[y[:, None], (np.arange(W)[None, :] + dx[:, None]) % W]
    a = out.astype(np.float32); a[..., 2] = np.roll(a[..., 2], int(4 * W / 1080), axis=1); a[..., 0] = np.roll(a[..., 0], -int(4 * W / 1080), axis=1)
    return np.clip(a * 1.08, 0, 255).astype(np.uint8)
