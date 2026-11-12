"""T9 · Dos Max a la vez (61,3–69,75 s): pantalla partida en diagonal.
Arriba (casa, de día): la copia se arma en la cabina de la cocina, sale feliz, saluda de lejos a la esposa y se va al cuarto.
Abajo (México, sin luz): Max sentado en el piso de la cabina con una vela; la ventanita pasa del día a la noche; bosteza.
67,6 "ding": vuelve la luz, él da un brinco y su mitad empuja a la otra hasta llenar la pantalla.
La entrada de la diagonal sobre la cara de la T8 (61,3–62,3) se arma en el montaje."""
import math, os, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, cabina as CB, arena as AR, actor_rb
import local_mx as LM, pantalla as PT
import recursos_vel as RV

T_ARMA, T_SALE, T_CAMINA, T_DING = 62.3, 64.1, 65.6, 67.6
ESPOSA9 = np.array([16.85, 7.75, C.Z0]); ESPOSA9_GIRO = -100.0
PASO = np.array([19.6, 8.0, C.Z0])
MX_PISO = LM.CABINA + [0, 0, CB.PISO_Z]
VELA = LM.CABINA + np.array([-.95, -.35, 0])
MIRA_ARRIBA = (15.45, 7.50, -.60)      # apunta bajo: la cabina y la esposa quedan en la mitad de ARRIBA, que es la que se ve


def diagonal(H, W, k):
    """Máscara: True = mitad de arriba. k = 0 la diagonal en el medio; k = 1 la de abajo ocupa todo; k = -1 la de arriba todo."""
    y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
    a = .60 - .22 * (x / W)                              # la línea baja de izquierda a derecha... sube hacia la derecha
    a = a + _desp(k)
    return y < a * H


def _desp(k):
    return k * .85 if k < 0 else -k * .85


class Toma:
    INI, FIN = 61.3, 69.75

    def preparar(s, E):
        s.E = E; C.preparar(E)
        s.fijos_casa = K.fijos(E)
        import esposa_tp
        s.esposa = esposa_tp.Esposa(E)
        s.A = AR.Arena(E, K.PISO, t_ini=T_ARMA, t_fin=T_SALE - .1, z_ini=0.0, z_fin=2.10, invertir=True)
        s.t_pre = T_ARMA
        import t08                                       # 61,3-62,3: abajo sigue la cara alumbrada por el celular (la T8)
        s.t8 = t08.Toma(); s.t8.E = E; s.t8.W, s.t8.H = E.r.w, E.r.h
        s.t8.pos_press, s.t8.yaw_press = CB.pose_presionar(LM.CABINA, LM.CAB_YAW)
        s.t8.pista = caras([(-9, 99, 'desaprobacion')])
        s.t8.cel_adel, s.t8.cel_hondo, s.t8.cel_lado, s.t8.cel_curva, s.t8.cel_pulgar = t08.CELULAR
        s.t8.cel_R = np.eye(3); s.t8._apoyar_celular()                  # el mismo agarre medido que en la T8
        s.p_copia = caras([(-9, 63.6, 'neutral'), (63.6, 99, 'contento')])
        s.p_mx = caras([(-9, 65.3, 'impaciente'), (65.3, T_DING, 'cansado'), (T_DING, 99, 'susto')], semilla=11)
        s.p_camina = colocar(E, 'g15_caminar', 1.0, K.PISO + K.F * .9, 0.0)
        d = PASO[:2] - (K.PISO + K.F * .9)[:2]; s.yaw_camina = math.degrees(math.atan2(d[0], -d[1]))
        s.p_camina = colocar(E, 'g15_caminar', 1.0, K.PISO + K.F * .9, s.yaw_camina)

    # ---------------------------------------------------------------- arriba: la casa
    def _copia(s, t):
        E = s.E
        if t < T_SALE:
            E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0} if t < 63.6 else {}
            return max_objs(E, 'g16_espera_respirando', 1 + (t - T_ARMA) * .6, K.PISO, K.YAW, pista=s.p_copia, t_escena=t, suelo=K.PISO[2])
        E.max.cara_manual = {}
        if t < T_CAMINA:
            return max_objs(E, 'g18_saludar', .3 + (t - T_SALE) * 1.2, K.PISO + K.F * .35, K.YAW + 25, pista=s.p_copia, t_escena=t, suelo=C.Z0)
        return max_objs(E, 'g15_caminar', 1.0 + (t - T_CAMINA) * 1.1, s.p_camina, s.yaw_camina, pista=s.p_copia, t_escena=t, suelo=C.Z0)

    def avanzar(s, t):
        if t < T_ARMA: return
        o, M, J = s._copia(t)
        if t < T_SALE: s.A.avanzar(t); o = s.A.cortar(s.E, o, M)
        s._arriba = (o, M, J)

    def _render_arriba(s, E, t):
        o, M, J = s._arriba
        if t >= T_SALE:                                              # se vuelve a posar (abajo se usó el mismo Max)
            o, M, J = s._copia(t)
        e = s.esposa.objetos(t, ESPOSA9, ESPOSA9_GIRO)
        objs = s.fijos_casa + o + e
        E.posts = [s.A.post] if t < T_SALE else []
        u = (t - s.INI) / (s.FIN - s.INI)                               # la cámara nunca queda quieta: se acerca despacio
        cam = (tuple(np.array((19.55, 4.15, 1.95)) + np.array((-.45, .35, -.10)) * u), MIRA_ARRIBA, 62 - 3 * u)
        return E.render(objs, cam, C.luz(cam[1], cam[0]))

    # ---------------------------------------------------------------- abajo: México sin luz
    def _mx(s, t):
        E = s.E
        if t < 65.3: n, tc = 'tp_sentado_bostezo_1', .2 + (t - T_ARMA) * .35
        elif t < T_DING: n, tc = 'tp_sentado_bostezo_1', 1.25 + (t - 65.3) * .9
        else: n, tc = 'tp_sentado_sobresalto_2', 1.0 + (t - T_DING) * 1.0
        E.max.cara_manual = {}
        return max_objs(E, n, tc, MX_PISO + np.array([-.08, 0, 0]), LM.CAB_YAW, pista=s.p_mx, t_escena=t, suelo=MX_PISO[2] if t < T_DING + .3 else MX_PISO[2])

    def _vela(s, t):
        u = min(1.0, max(0.0, (t - T_ARMA) / (T_DING - T_ARMA)))
        alto = .30 - .21 * u
        o = RV.modelo(s.E.r, 'tp_velas', tuple(VELA), 0.0, alto=alto)
        top = VELA + (0, 0, alto + .02)
        Mf = np.eye(4, dtype=np.float32); fl = 1 + .15 * math.sin(t * 23) + .1 * math.sin(t * 37)
        Mf[:3, :3] = np.diag([.012, .012, .035 * fl]); Mf[:3, 3] = top
        o.append(dict(mesh='box', model=Mf, texture='white', color=(1.0, .75, .35), emission=9, noshadow=True))
        return o, top

    def _render_abajo(s, E, t):
        o, M, J = s._mx(t)
        ventana = 1.0 - min(1.0, max(0.0, (t - 63.0) / (T_DING - 63.5)))
        luces = [0.0] * 5 if t < T_DING else [1.0] * 5
        vela, top = s._vela(t)
        objs = LM.escena(E, t, luces, ventana=ventana, luz_cabina=0.0 if t < T_DING else 1.0) + o + (vela if t < T_DING + .2 else [])
        extra = [dict(pos=tuple(top + (0, 0, .08)), col=(.9 * (1 + .1 * math.sin(t * 19)), .55, .22))] if t < T_DING + .2 else []
        h = J['Hips']
        u = (t - s.INI) / (s.FIN - s.INI)
        w = suave((t - T_DING) / .9)                                    # al volver la luz la pantalla es toda suya: lo centra y lo sigue
        cam = (tuple(np.array([.15, -1.85, 1.05]) + np.array([.35, .30, -.05]) * u + np.array([0, -.5, .25]) * w), tuple(mezcla(h + (-.45, .20, .75), J['Head'] * .5 + h * .5 + (0, 0, .05), w)), 54 - 4 * u)
        E.posts = []
        Lz = LM.luz(cam[1], cam[0], luces, ventana * .6, extra)
        if t < T_DING: Lz.update(env_k=.05 + .2 * ventana)
        return E.render(objs, cam, Lz)

    def _arriba_antes(s, E, t):
        """61,3-62,3: la casa con la cabina todavía vacía (la copia empieza a armarse en 62,3)."""
        e = s.esposa.objetos(t, ESPOSA9, ESPOSA9_GIRO)
        cam = ((19.55, 4.15, 1.95), MIRA_ARRIBA, 62); E.posts = []
        return E.render(s.fijos_casa + e, cam, C.luz(cam[1], cam[0]))

    def cuadro(s, E, t):
        if t >= T_DING + .7:
            return s._render_abajo(E, t)
        if t < T_ARMA:                                   # la diagonal cae como una cortina sobre la cara de la T8
            a = s._arriba_antes(E, t)
            guard = (E.cielo, getattr(E, 'zsuelo', 0.0), getattr(E, 'calido', 0.0), getattr(E, 'grado', {}))
            b = s.t8.cuadro(E, max(t, 60.5))
            E.cielo, E.zsuelo, E.calido, E.grado = guard
            k = -1.0 + suave((t - s.INI) / (T_ARMA - s.INI))
        else:
            a = s._render_arriba(E, t); b = s._render_abajo(E, t)
            k = 0.0 if t < T_DING else min(1.0, ((t - T_DING) / .7) ** 1.5)
        H, W = a.shape[:2]
        m = diagonal(H, W, k)
        img = np.where(m[..., None], a, b)
        # borde fino de la diagonal
        y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
        a_ = (.60 - .22 * (x / W) + _desp(k)) * H
        borde = np.abs(y - a_) < max(2, H // 320)
        img[np.broadcast_to(borde, (H, W))] = 235
        return img
