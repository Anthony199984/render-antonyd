"""T2 · Rumbo a Tokio (6,2–12,3 s). El rebobinado 6,2–7,0 se arma en el montaje con los cuadros de la T1 al revés.
7,0  travelling lateral: Max entra a la cabina frotándose las manos ("en la cabina de tu casa")
8,55 sobre el hombro: el dedo toca la pantalla, el globo gira y se ilumina Japón ("eliges Japón como destino")
10,0 contraplano: la esposa saluda sin levantar la vista del café ("y mientras te despides de tu esposa")
11,65 plano medio: Max guiña y levanta el pulgar."""
import math, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, cabina as CB, pantalla as PT, actor_rb, ik_tp as IK
import recursos_vel as RV

import os
CAMINAR = os.environ.get('CAMINAR', 'tp_caminar_frotar_3')
TOKIO = (35.68, 139.69); CASA = (19.45, -70.70)
T_TOQUE = 8.85
PULGAR = os.environ.get('PULGAR', 'tp_pulgar_pecho_2')        # Kimodo no hace el pulgar: guiño + mano arriba de despedida


class Toma:
    INI, FIN = 6.3, 12.3

    def preparar(s, E):
        C.preparar(E); s.E = E
        s.fijos = K.fijos(E, pantalla='cab_pant_t2')
        import esposa_tp
        s.esposa = esposa_tp.Esposa(E)
        s.pista = caras([(-9, 8.55, 'euforia'), (8.55, 10.0, 'contento'), (10.0, 99, 'simpatico')])
        s.pos_press, s.yaw_press = CB.pose_presionar(K.CAB, K.YAW)
        c = clip(E, CAMINAR); ix = c.idx
        P0 = c.globales(0)[1][ix['Hips']]; P1 = c.globales(c.dur)[1][ix['Hips']]
        s.d_camina = float(np.hypot(P1[0] - P0[0], P1[2] - P0[2]))
        s.dur_camina = c.dur

    # ---------------------------------------------------------------- pantalla
    def _pantalla(s, t):
        if t < T_TOQUE: im = PT.globo(CASA[1], 18)
        else:
            u = suave((t - T_TOQUE) / .55)
            lon = CASA[1] + (TOKIO[1] - CASA[1]) * u; lat = 18 + (30 - 18) * u
            im = PT.globo(lon, lat, TOKIO, brillo=max(0.0, (t - T_TOQUE - .45) / .6))
        PT.poner(s.E.r, 'cab_pant_t2', im)

    # ---------------------------------------------------------------- Max
    def _max(s, t):
        E = s.E
        if t < 8.55:                                   # camina hacia la cabina a velocidad real (desde donde hizo el gancho)
            return K.caminar(E, t, s.pista)
        if t < 11.65:
            tc = .85 + (t - T_TOQUE)
            if not hasattr(s, 'M_press'):
                s.M_press = max_objs(E, 'tp_presionar_boton_1', 1.0, s.pos_press, s.yaw_press, suelo=K.PISO[2])[1]
                sp, n = CB.pantalla_pos(K.CAB, K.YAW); s.p_toque = sp + n * .006
            w = suave((t - T_TOQUE + .30) / .30) * (1 - suave((t - T_TOQUE - .75) / .35))
            return max_objs(E, 'tp_presionar_boton_1', min(tc, 2.6), s.pos_press, s.yaw_press, pista=s.pista, t_escena=t, suelo=K.PISO[2],
                            ik=IK.tocar(s.M_press, s.p_toque, w))
        tc = .2 + (t - 11.65) * 1.0
        E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeSquintLeft': .6, 'mouthSmileLeft': .5} if 11.85 < t < 12.15 else {}
        return max_objs(E, PULGAR, tc, K.PISO, K.YAW, pista=s.pista, t_escena=t, suelo=K.PISO[2])

    def _esposa(s, t):
        return s.esposa.objetos(t, K.ESPOSA, K.ESPOSA_GIRO, saluda=(10.15, 11.75))

    # ---------------------------------------------------------------- cámara
    def camara(s, t, J):
        if t < 8.55:
            u = (t - 6.3) / 2.25; h = J['Hips']
            lado = -K.S * 2.4 + K.F * .2
            w = suave((u - .40) / .50)                       # al final pasa al frente de la puerta: nunca mira a través del vidrio de la cabina
            ojo = mezcla(h + lado + (0, 0, .25) + K.F * (.3 - .6 * u), K.PISO + K.F * 3.1 - K.S * 1.0 + (0, 0, 1.35), w)
            return (tuple(ojo), tuple(h + (0, 0, .2) - K.F * .25 * (1 - w)), 46)
        if t < 10.0:
            sp, n = CB.pantalla_pos(K.CAB, K.YAW)          # n apunta de la pantalla hacia Max
            u = (t - 8.55) / 1.45
            lat = -np.cross(n, [0, 0, 1.0])                   # hacia la derecha de Max (la mano que toca)
            ojo = sp + n * .44 - lat * .40 + (0, 0, .12 - .03 * u)                # cerca de la pantalla: el dedo y el globo
            return (tuple(ojo), tuple(sp + n * .12 + (0, 0, -.06)), 40 - 7 * u)
        if t < 11.65:
            u = (t - 10.0) / 1.65; e = K.ESPOSA
            ojo = K.PISO + K.F * .35 + (0, 0, 1.55) + K.S * -.15
            mira = e + (0, 0, 1.25)
            ojo = mezcla(ojo, mira, .40 + .06 * u)             # más abierto: entra la mano que saluda
            return (tuple(ojo), tuple(mira + (0, 0, .03)), 36)
        u = (t - 11.65) / .65; h = J['Head']
        ojo = h + K.F * 1.95 + K.S * .70 + (0, 0, -.12)
        return (tuple(ojo + K.F * -.15 * u), tuple(h + (0, 0, -.12)), 42)

    def cuadro(s, E, t):
        s._pantalla(t)
        o, M, J = s._max(t)
        objs = s.fijos + o
        if t >= 9.9: objs += s._esposa(t)
        cam = s.camara(t, J)
        E.posts = []
        return E.render(objs, cam, C.luz(cam[1], cam[0]))
