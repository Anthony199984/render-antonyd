"""T10 · La cama (69,75–73,36 s), el dormitorio de noche con las lámparas.
69,75 subjetiva desde la puerta, empujando despacio: la esposa lee en la cama ("y al llegar te encuentras a tu esposa")
71,15 paneo: descubre al otro Max dormido con la boca abierta; dolly zoom a la cara del que llega, se le cae la bolsa ("en la cama")
72,30 el dormido se incorpora; los dos se miran; la esposa mira a uno, al otro y sigue leyendo ("con el otro tú")."""
import math, numpy as np
from tp_base import *
import casa_lovely as C, actor_rb
import recursos_vel as RV

T_DOS = 70.85                 # plano claro de la cama: ella y EL OTRO MAX juntos ("en la cama, con el otro tú")
T_CARA, T_SUELTA, T_DESPIERTA = 71.95, 72.20, 72.50
COLCHON = C.altura_colchon(25.0)
ESPOSA10 = np.array([25.66, 6.72, COLCHON - .03]); ESPOSA10_GIRO = -90.0   # recostada en la cabecera (X = 26,5)
COPIA = np.array([25.25, 5.62, 0.0]); COPIA_YAW = -90.0           # acostado: la cabeza hacia la cabecera (+X)
LLEGA = np.array([22.15, 7.55, C.Z0]); LLEGA_YAW = 72.0
LAMPARAS = [np.array([26.45, 7.55, 1.05]), np.array([26.45, 4.85, 1.05])]


def _malla_almohada(r):
    """Almohada redondeada (superelipsoide) de 1 x 1 x 1: se escala con la matriz. Bordes blandos, centro abultado."""
    if 'almohada_tp' in r.meshes: return
    import lovely_vida as V
    Mx = V.Malla(); nu, nv = 28, 16
    def pt(u, v):
        cu, su, cv, sv = math.cos(u), math.sin(u), math.cos(v), math.sin(v)
        f = lambda c, e: math.copysign(abs(c) ** e, c)
        return np.array([.5 * f(cv, .9) * f(cu, .35) * .9 + 0, .5 * f(cv, .35) * f(su, .35), .5 * f(sv, .35)])
    for i in range(nu):
        for j in range(nv):
            u0, u1 = 2 * math.pi * i / nu, 2 * math.pi * (i + 1) / nu; v0, v1 = -math.pi / 2 + math.pi * j / nv, -math.pi / 2 + math.pi * (j + 1) / nv
            a, b, c, d = pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)
            Mx.quad(a, b, c, d, (1, 1, 1))
    Mx.subir(r, 'almohada_tp')


class Toma:
    INI, FIN = 69.75, 73.36

    def preparar(s, E):
        C.preparar(E); s.E = E
        s.fijos = C.escena(E, 0.0, cuarto=True, cocina=True)
        s.esposa = actor_rb.ActorRB(E.r, 'rb_mujer_f08', 'esposa', 1.64)
        s.c_lee = Clip(ruta_anim('tp_esposa_cama_leer_1'))
        s.p_llega = caras([(-9, 71.85, 'contento'), (71.85, 99, 'shock')])
        s.p_copia = caras([(-9, T_DESPIERTA + .1, 'neutral'), (T_DESPIERTA + .1, 99, 'confundido')], semilla=5)
        s.pos_llega = colocar(E, 'tp_bolsa_soltar_2', 2.2, LLEGA, LLEGA_YAW)
        s.dz_copia = None
        _malla_almohada(E.r)
        o, M, J = s._llega(T_SUELTA)                                  # dónde queda la bolsa al soltarla (sin estado entre trozos)
        W = mano_mat(E, M, 'Right'); s.bolsa = (W[:3, 3] + np.array([0, 0, -.30]), T_SUELTA)

    def _llega(s, t):
        if t >= T_DESPIERTA:                                           # señala al otro (corte: plano por encima del hombro)
            if not hasattr(s, 'pos_senala'):
                o, M, J = max_objs(s.E, 'tp_bolsa_soltar_2', 4.9, s.pos_llega, LLEGA_YAW, suelo=C.Z0)
                s.pos_senala = colocar(s.E, 'tp_senalar_frente_1', 0.0, J['Hips'], LLEGA_YAW - 12)
            return max_objs(s.E, 'tp_senalar_frente_1', .3 + (t - T_DESPIERTA) * 1.6, s.pos_senala, LLEGA_YAW - 12, pista=s.p_llega, t_escena=t, suelo=C.Z0)
        tc = min(1.0 + (t - s.INI) * .90, 4.9)                        # suelta la bolsa en T_SUELTA
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
        """Lee recostada en la cabecera: el cuerpo entero se inclina hacia atrás hasta que las piernas quedan acostadas en el
        colchón (medido), y se apoya en una almohada. El libro va entre las dos manos."""
        A = s.esposa
        tc = (t - s.INI) * 1.0 + .5
        if t > T_DESPIERTA: tc = 2.4 + (t - T_DESPIERTA) * 1.4          # mira a un lado y al otro
        tc = min(tc, s.c_lee.dur - .05)
        if not hasattr(s, 'M_esp'):
            A.objetos(s.c_lee, .5, tuple(ESPOSA10), ESPOSA10_GIRO)
            cad = A.hueso('Bip01 Pelvis'); pie = (A.hueso('Bip01 L Foot') + A.hueso('Bip01 R Foot')) / 2
            phi = math.atan2(cad[2] - pie[2] - .04, np.hypot(*(cad[:2] - pie[:2])))        # cuánto cuelgan las piernas
            c, sn = math.cos(-phi), math.sin(-phi)                                     # giro sobre el eje Y del mundo (ella mira a -X): se echa atrás
            R = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])
            T1 = np.eye(4); T1[:3, 3] = -cad; R4 = np.eye(4); R4[:3, :3] = R; T2 = np.eye(4); T2[:3, 3] = cad
            s.M_esp = T2 @ R4 @ T1; s.phi = phi
            A.objetos(s.c_lee, .5, tuple(ESPOSA10), ESPOSA10_GIRO, M_extra=s.M_esp)
            s.M_esp[2, 3] += COLCHON - .015 - A.ult_zmin                                # apoyada en el colchón
        o = A.objetos(s.c_lee, tc, tuple(ESPOSA10), ESPOSA10_GIRO, cara='neutral', M_extra=s.M_esp, dedos={'L': 30, 'R': 30})
        a, b = A.hueso('Bip01 L Finger2'), A.hueso('Bip01 R Finger2'); c = (a + b) / 2
        x = b - a; x /= np.linalg.norm(x); h = A.hueso('Bip01 Head'); z = h - c; z = z - x * (z @ x); z /= np.linalg.norm(z); y = np.cross(z, x)
        R = np.stack([x, y, z], 1)                                                     # el libro abierto mirando a su cara
        for tam, col, off in (((.26, .19, .018), (.55, .10, .08), (0, .02, -.012)), ((.25, .18, .012), (.95, .93, .86), (0, .02, .0))):
            Mb = np.eye(4, dtype=np.float32); Mb[:3, :3] = R @ np.diag(tam); Mb[:3, 3] = c + R @ np.array(off)
            o.append(dict(mesh='box', model=Mb, texture='white', color=col, roughness=.7))
        o.append(dict(mesh='almohada_tp', model=s._almohada(tc), texture='white', color=(.94, .94, .96), roughness=.85))
        return o

    def _almohada(s, tc_ahora):
        """Almohada blanda entre la espalda y la cabecera, MEDIDA con la malla de ella en toda la toma: su cara de adelante
        queda 1 cm detrás de la espalda (nunca la atraviesa) y la de atrás contra la cabecera."""
        if getattr(s, 'Mp', None) is not None: return s.Mp
        A = s.esposa; Vs = []
        for t in np.linspace(s.INI, s.FIN, 9):
            tc = (t - s.INI) * 1.0 + .5
            if t > T_DESPIERTA: tc = 2.4 + (t - T_DESPIERTA) * 1.4
            A.objetos(s.c_lee, min(tc, s.c_lee.dur - .05), tuple(ESPOSA10), ESPOSA10_GIRO, M_extra=s.M_esp); Vs.append(A.ult_V)
        V = np.concatenate(Vs)
        esp = A.hueso('Bip01 Spine1')
        ca, sa = math.cos(-.55), math.sin(-.55); Rp = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])     # inclinada contra la cabecera
        T_, AN, AL = .15, .62, .40                                                     # grosor, ancho, alto
        cen = np.array([esp[0], esp[1], esp[2] - .04])
        L = (V - cen) @ Rp                                                             # en ejes de la almohada (x = grosor, hacia la cabecera)
        dentro = (np.abs(L[:, 1]) < AN / 2 + .02) & (np.abs(L[:, 2]) < AL / 2 + .02)
        x0 = float(L[dentro, 0].max()) + T_ / 2 + .01 if dentro.any() else .17
        c = cen + Rp @ np.array([x0, 0, 0])
        Mp = np.eye(4, dtype=np.float32); Mp[:3, :3] = Rp @ np.diag([T_, AN, AL])
        from scipy.spatial import cKDTree                                              # medido con la forma real (redondeada)
        f = lambda q, e: np.sign(q) * np.abs(q) ** e
        uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 240), np.linspace(-np.pi / 2, np.pi / 2, 120))
        Pu = np.stack([.45 * f(np.cos(vv), .9) * f(np.cos(uu), .35), .5 * f(np.cos(vv), .35) * f(np.sin(uu), .35), .5 * f(np.sin(vv), .35)], -1).reshape(-1, 3)
        Vc = V[np.linalg.norm(V - c, axis=1) < .6]
        for _ in range(40):
            Pw = Pu @ Mp[:3, :3].T + c; dist, i = cKDTree(Pw).query(Vc)
            mete = (np.linalg.norm(Vc - c, axis=1) < np.linalg.norm(Pw[i] - c, axis=1))
            s.almohada_mete = float(dist[mete].max()) if mete.any() else 0.0
            if s.almohada_mete < .004: break
            c = c + Rp[:, 0] * .006                                                     # hacia la cabecera, de a 6 mm
        Mp[:3, 3] = c
        s.Mp = Mp; s.almohada_hueco = float(x0 - T_ / 2 - L[dentro, 0].max()) if dentro.any() else None
        return Mp

    def cuadro(s, E, t):
        o2, M2, J2 = s._copia(t)
        o1, M1, J1 = s._llega(t)
        subjetiva = False
        objs = s.fijos + ([] if subjetiva else o1) + o2 + s._esposa(t) + ([] if subjetiva else s._bolsa(t, M1))
        if subjetiva: s._bolsa(t, M1)
        for p in LAMPARAS:                                             # pantallas de las lámparas encendidas
            Ml = np.eye(4, dtype=np.float32); Ml[:3, :3] = np.diag([.16, .16, .18]); Ml[:3, 3] = p
            objs.append(dict(mesh='box', model=Ml, texture='white', color=(1.0, .82, .55), emission=2.2, noshadow=True))
        ec = s.esposa.hueso('Bip01 Head')
        cp = J2['Head']
        if t < T_DOS:                                                  # desde la puerta: ella lee en la cama
            u = (t - s.INI) / (T_DOS - s.INI)
            ojo = np.array([21.75, 8.55, 1.72]) + np.array([.45, -.30, 0]) * suave(u)
            cam = (tuple(ojo), tuple(ec + (0, .1, -.25)), 34)
        elif t < T_CARA:                                               # desde los pies de la cama, bajando: ella y el otro Max dormido, bien claros
            u = suave((t - T_DOS) / (T_CARA - T_DOS))
            ojo = mezcla(np.array([23.55, 4.70, 2.50]), np.array([23.95, 4.80, 2.35]), u)        # en diagonal desde la esquina de los pies: caben los dos
            mira = (ec + cp) / 2 + (-.15, 0, -.22)
            cam = (tuple(ojo), tuple(mira), 58 - 6 * u)
        elif t < T_DESPIERTA:                                           # dolly zoom a la cara del que llega; se le cae la bolsa
            v = suave((t - T_CARA) / (T_DESPIERTA - T_CARA)); h = J1['Head']; d = 1.7 - .8 * v
            fov = 2 * math.degrees(math.atan(math.tan(math.radians(14)) * 1.7 / d))
            dirc = np.array([.95, -.30, -.05]); dirc /= np.linalg.norm(dirc)
            cam = (tuple(h + dirc * d), tuple(h + (0, 0, -.06)), fov)
        else:                                                           # por encima del hombro del que llega: el otro se incorpora y lo mira
            u = (t - T_DESPIERTA) / (s.FIN - T_DESPIERTA)
            cam = ((21.00 + .2 * u, 7.78, 2.42), tuple(cp * .6 + ec * .4 + (-.2, 0, -.15)), 42 - 3 * u)
        E.posts = []
        Lz = C.luz(cam[1], cam[0])
        Lz.update(sun_col=(.02, .025, .04), env_top=(.05, .06, .10), env_hor=(.05, .05, .08), env_bot=(.03, .03, .04), env_k=1.1,
                  bg_top=(.01, .015, .04), bg_hor=(.02, .03, .07), bg_bot=(.01, .01, .02))
        Lz['cielo'] = False
        Lz['lights'] = [dict(pos=tuple(p + (0, 0, .05)), col=(3.4, 2.5, 1.5)) for p in LAMPARAS] + [dict(pos=(24.5, 6.1, 2.45), col=(2.6, 2.3, 1.9)), dict(pos=(23.4, 6.6, 2.5), col=(1.2, 1.1, 1.0)), dict(pos=(21.3, 7.2, 2.4), col=(.9, .85, .8))]
        return E.render(objs, cam, Lz)
