"""T10 · La cama (69,75–73,36 s), el dormitorio de noche con las lámparas.
69,75 subjetiva desde la puerta, empujando despacio: la esposa lee en la cama ("y al llegar te encuentras a tu esposa")
71,15 paneo: descubre al otro Max dormido con la boca abierta; dolly zoom a la cara del que llega, se le cae la bolsa ("en la cama")
72,30 el dormido se incorpora; los dos se miran; la esposa mira a uno, al otro y sigue leyendo ("con el otro tú")."""
import math, numpy as np
from tp_base import *
import casa_lovely as C, actor_rb
import recursos_vel as RV

T_PANEO, T_DESPIERTA = 71.15, 72.30
T_SUELTA = 71.45
COLCHON = C.altura_colchon(25.0)
ESPOSA10 = np.array([25.75, 6.72, COLCHON - .03]); ESPOSA10_GIRO = -90.0
COPIA = np.array([25.25, 5.62, 0.0]); COPIA_YAW = -90.0           # acostado: la cabeza hacia la cabecera (+X)
LLEGA = np.array([21.05, 7.70, C.Z0]); LLEGA_YAW = 72.0
LAMPARAS = [np.array([26.45, 7.55, 1.05]), np.array([26.45, 4.85, 1.05])]


class Toma:
    INI, FIN = 69.75, 73.36

    def preparar(s, E):
        C.preparar(E); s.E = E
        s.fijos = C.escena(E, 0.0, cuarto=True, cocina=True)
        s.esposa = actor_rb.ActorRB(E.r, 'rb_mujer_f08', 'esposa', 1.64)
        s.c_lee = Clip(ruta_anim('tp_esposa_cama_leer_1'))
        s.p_llega = caras([(-9, 71.3, 'contento'), (71.3, 99, 'shock')])
        s.p_copia = caras([(-9, 72.4, 'neutral'), (72.4, 99, 'confundido')], semilla=5)
        s.pos_llega = colocar(E, 'tp_bolsa_soltar_2', 2.2, LLEGA, LLEGA_YAW)
        s.dz_copia = None
        o, M, J = s._llega(T_SUELTA)                                  # dónde queda la bolsa al soltarla (sin estado entre trozos)
        W = mano_mat(E, M, 'Right'); s.bolsa = (W[:3, 3] + np.array([0, 0, -.30]), T_SUELTA)

    def _llega(s, t):
        if t >= T_DESPIERTA:                                           # señala al otro (corte: plano por encima del hombro)
            if not hasattr(s, 'pos_senala'):
                o, M, J = max_objs(s.E, 'tp_bolsa_soltar_2', 4.9, s.pos_llega, LLEGA_YAW, suelo=C.Z0)
                s.pos_senala = colocar(s.E, 'tp_senalar_frente_1', 0.0, J['Hips'], LLEGA_YAW - 12)
            return max_objs(s.E, 'tp_senalar_frente_1', .3 + (t - T_DESPIERTA) * 1.6, s.pos_senala, LLEGA_YAW - 12, pista=s.p_llega, t_escena=t, suelo=C.Z0)
        tc = min(1.0 + (t - s.INI) * 1.05, 4.9)
        return max_objs(s.E, 'tp_bolsa_soltar_2', tc, s.pos_llega, LLEGA_YAW, pista=s.p_llega, t_escena=t, suelo=C.Z0)

    def _copia(s, t):
        E = s.E
        if s.dz_copia is None:                                        # acostado SOBRE el colchón (medido con la malla)
            o, M, J = max_objs(E, 'tp_acostado_incorporarse_2', 0.0, COPIA, COPIA_YAW, suelo=0.0)
            P = np.concatenate([p for p, _ in E.max._ult_pose[2]]); z = (P @ M[:3, :3].T + M[:3, 3])[:, 2]
            s.dz_copia = COLCHON - .025 - float(z.min())
        tc = 0.0 if t < T_DESPIERTA else min(1.0 + (t - T_DESPIERTA) * 1.9, 2.95)
        E.max.cara_manual = {'jawOpen': .55} if t < T_DESPIERTA else ({'eyeBlinkLeft': 1.0} if t < T_DESPIERTA + .25 else {})
        o, M, J = max_objs(E, 'tp_acostado_incorporarse_2', tc, COPIA, COPIA_YAW, pista=s.p_copia, t_escena=t, suelo=s.dz_copia, otro=True)
        E.max.cara_manual = {}
        return o, M, J

    def _bolsa(s, t, M):
        """La bolsa de tacos en la mano derecha hasta que la suelta; después cae al piso y se queda."""
        E = s.E
        if t < T_SUELTA:
            W = mano_mat(E, M, 'Right'); p = W[:3, 3] + np.array([0, 0, -.30])
        else:
            p0, t0 = s.bolsa; dt = t - T_SUELTA
            p = p0.copy(); p[2] = max(C.Z0, p0[2] - 4.9 * dt * dt)
        return RV.modelo(E.r, 'tp_bolsa', tuple(p), LLEGA_YAW, alto=.30)

    def _esposa(s, t):
        tc = (t - s.INI) * 1.25 + .5
        if t > T_DESPIERTA: tc = 2.4 + (t - T_DESPIERTA) * 1.4          # mira a un lado y al otro
        if not hasattr(s, 'dz_esposa'):                                 # sentada SOBRE el colchón (medido con la malla)
            s.esposa.objetos(s.c_lee, .5, tuple(ESPOSA10), ESPOSA10_GIRO); s.dz_esposa = COLCHON - .02 - s.esposa.ult_zmin
        o = s.esposa.objetos(s.c_lee, min(tc, s.c_lee.dur - .05), tuple(ESPOSA10 + (0, 0, s.dz_esposa)), ESPOSA10_GIRO, cara='neutral')
        a, b = s.esposa.hueso('Bip01 L Hand'), s.esposa.hueso('Bip01 R Hand'); c = (a + b) / 2
        x = b - a; x /= np.linalg.norm(x); z = np.array([0, 0, 1.0]); y = np.cross(z, x); y /= np.linalg.norm(y); z = np.cross(x, y)
        R = np.stack([x, y, z], 1)
        for tam, col, off in (((.24, .17, .02), (.55, .10, .08), (0, 0, -.01)), ((.23, .16, .012), (.95, .93, .86), (0, 0, .006))):
            Mb = np.eye(4, dtype=np.float32); Mb[:3, :3] = R @ np.diag(tam); Mb[:3, 3] = c + R @ np.array(off) + (0, 0, .03)
            o.append(dict(mesh='box', model=Mb, texture='white', color=col, roughness=.7))
        return o

    def cuadro(s, E, t):
        o2, M2, J2 = s._copia(t)
        o1, M1, J1 = s._llega(t)
        subjetiva = t < T_PANEO + .45
        objs = s.fijos + ([] if subjetiva else o1) + o2 + s._esposa(t) + ([] if subjetiva else s._bolsa(t, M1))
        if subjetiva: s._bolsa(t, M1)
        for p in LAMPARAS:                                             # pantallas de las lámparas encendidas
            Ml = np.eye(4, dtype=np.float32); Ml[:3, :3] = np.diag([.16, .16, .18]); Ml[:3, 3] = p
            objs.append(dict(mesh='box', model=Ml, texture='white', color=(1.0, .82, .55), emission=2.2, noshadow=True))
        ojo_sub = J1['Head'] + np.array([.12, -.04, -.02])                 # sus ojos (no se dibuja a sí mismo)
        if t < T_PANEO:
            u = (t - s.INI) / (T_PANEO - s.INI)
            cam = (tuple(ojo_sub), tuple(ESPOSA10 + (0, .1, .75)), 50)
        elif t < T_DESPIERTA:
            u = (t - T_PANEO) / (T_DESPIERTA - T_PANEO)
            if u < .4:                                                  # paneo a la copia
                v = suave(u / .4); cam = (tuple(ojo_sub), tuple(mezcla(ESPOSA10 + (0, .1, .75), J2['Head'], v)), 50)
            else:                                                       # dolly zoom a la cara del que llega
                v = suave((u - .4) / .6); h = J1['Head']; d = 1.6 - .8 * v
                fov = 2 * math.degrees(math.atan(math.tan(math.radians(14)) * 1.6 / d))
                dirc = np.array([.95, -.30, -.05]); dirc /= np.linalg.norm(dirc)
                cam = (tuple(h + dirc * d), tuple(h + (0, 0, -.05)), fov)
        else:
            u = (t - T_DESPIERTA) / (s.FIN - T_DESPIERTA)
            h = J1['Head']; ob = (J2['Head'] + ESPOSA10 + (0, 0, .9)) / 2
            cam = (tuple(h + np.array([-.55, .42, .12]) + np.array([.05, 0, 0]) * u), tuple(ob + (0, 0, -.1)), 52)
        E.posts = []
        Lz = C.luz(cam[1], cam[0])
        Lz.update(sun_col=(.02, .025, .04), env_top=(.05, .06, .10), env_hor=(.05, .05, .08), env_bot=(.03, .03, .04), env_k=.35,
                  bg_top=(.01, .015, .04), bg_hor=(.02, .03, .07), bg_bot=(.01, .01, .02))
        Lz['cielo'] = False
        Lz['lights'] = [dict(pos=tuple(p + (0, 0, .05)), col=(2.6, 1.9, 1.1)) for p in LAMPARAS] + [dict(pos=(23.2, 6.5, 2.6), col=(.45, .40, .35))]
        return E.render(objs, cam, Lz)
