"""T6 · Dubái: el horno (tiempo viejo 36,3–43,7 s), 7 de la tarde todavía claro.
36,30 desde dentro de la cabina: se abre al aire y todo se vuelve blanco, con el aire que tiembla ("pero el aire del desierto")
37,30 el termómetro de la calle marca 46 ("te golpea como un horno")
38,30 órbita: Max frente al carrito, se abanica con la derecha y tiene el shawarma en la izquierda, sudando ("y empapado de sudor")
40,30 muerde el shawarma (la punta llega a la boca): asco ("buscas un lugar fresco")
41,80 lo deja en el mostrador delante del vendedor y sale corriendo a la cabina ("unos tacos en México")."""
import math, os, numpy as np
from tp_base import *
import dubai as DB, cabina as CB, actor_rb, sudor, ik_tp as IK
import recursos_vel as RV

T_TERMO, T_ORBITA, T_MUERDE, T_DEVUELVE, T_CORRE = 37.3, 38.3, 40.3, 41.8, 42.75
T_BOCA0, T_BOCA1, T_BAJA = 40.50, 41.20, 41.65            # sube a la boca / muerde hasta / ya lo bajó
T_DEJA0, T_DEJA1 = 41.95, 42.42                           # lleva el shawarma al mostrador / lo suelta
CAB = DB.CABINA; CAB_YAW = 90.0
PISO = CAB + [0, 0, CB.PISO_Z]
MAX_P = np.array([16.02, .78, L.ACERA_Z]); MAX_YAW = 90.0             # frente al carrito (mira +X), el vendedor al otro lado
VEND = np.array([17.62, 1.40, L.ACERA_Z]); VEND_GIRO = -90.0
TERMO = np.array([18.6, -.75, L.ACERA_Z])
MOSTRADOR = np.array([16.48, .80, 1.262])                              # donde deja el shawarma (encima del carrito)
BASE = 'tp_abanicarse_1'
LARGO = .25                                                            # el shawarma
ZC = 2.459 / 17.181 * LARGO / 2                                        # medio grosor


class Toma:
    INI, FIN = 36.3, 43.7

    def preparar(s, E):
        DB.preparar(E); s.E = E
        s.fijos = DB.escena(E, 0.0, cabina=False) + CB.objetos(E, CAB, CAB_YAW)
        s.vend = actor_rb.ActorRB(E.r, 'rb_Male_Adult_15', 'vend', 1.72)
        s.c_vend = Clip(ruta_anim('tp_vendedor_cortar_asador_1'))
        s.pista = caras([(-9, 37.3, 'susto'), (37.3, T_BOCA1 - .1, 'cansado'), (T_BOCA1 - .1, 42.4, 'asco'), (42.4, 99, 'decidido')])
        s.sud = sudor.Sudor(E)
        o, s.M0, J = max_objs(E, BASE, 1.0, MAX_P, MAX_YAW, suelo=L.ACERA_Z)        # la matriz mundo de Max en su sitio (no cambia)

    # ---------------------------------------------------------------- el brazo izquierdo con el shawarma
    def _ik(s, t):
        E = s.E; M0 = s.M0

        def fn(m, G, Pm, b):
            U, Fw, Rt = IK.ejes(m, G, b, 'Chest'); pc = Pm[m.idx['Chest']]
            F0 = IK.marco0(m, b, 'Left'); pal = IK.palma_local('Left')
            # 1) sostenerlo delante del pecho, derecho
            Fa = IK.marco_de(U * .95 + Fw * .30, Rt * .8 - Fw * .45)
            Wa = pc + Fw * .30 - Rt * .17 - U * .10
            # 2) la punta en la boca
            Uh, Fh, Rh = IK.ejes_cabeza(m, G, b); Pb, nb = IK.punto_cara(m, G, Pm, b, 'boca')
            Fb = IK.marco_de(-Fh * .72 + Uh * .62 + Rh * .22, Rh * .85 - Fh * .40)
            punta = pal + np.array([0, .05 + LARGO / 2, 0])
            Wb = Pb + nb * .012 - Fb @ punta
            # 3) acostado en el mostrador
            pm = IK.de_mundo(m, M0, MOSTRADOR + (0, 0, ZC + .012)); xm = IK.dir_de_mundo(M0, (0, -1.0, 0)); up = IK.dir_de_mundo(M0, (0, 0, 1.0))
            Fc = IK.marco_de(xm, -up)                                   # la palma hacia abajo, el shawarma a lo largo del carrito
            Wc = pm - Fc @ (pal + np.array([0, .05, 0]))
            ub = suave((t - T_BOCA0) / .42) * (1 - suave((t - T_BOCA1) / (T_BAJA - T_BOCA1)))
            uc = suave((t - T_DEJA0) / (T_DEJA1 - T_DEJA0))
            W = Wa * (1 - ub) + Wb * ub; F = IK._mez(Fa, Fb, ub)
            if ub > 0 and ub < 1: W = W + Fw * .10 * math.sin(ub * math.pi)      # el camino a la boca sale un poco hacia adelante
            if uc > 0:
                W = W * (1 - uc) + Wc * uc + U * .12 * math.sin(uc * math.pi); F = IK._mez(F, Fc, uc)
            w = 1.0 if t < T_DEJA1 + .1 else 1 - suave((t - T_DEJA1 - .1) / .3)
            if t >= T_MUERDE: IK.brazo_de(m, G, Pm, b, 'Right', clip(E, 'g16_espera_respirando'), 1.0)     # deja de abanicarse
            polo = -U * .8 - Rt * .55 - Fw * .2
            s.falta = IK.brazo(m, G, Pm, b, 'Left', W, F, polo=polo, w=w, giro_ante=.5)
            sujeto = t < T_DEJA1
            IK.dedos(m, G, Pm, b, 'Left', {d: (48, 52, 38) for d in IK.DEDOS} if sujeto else {d: (12, 14, 10) for d in IK.DEDOS},
                     pulgar=(38, 12, 22, 25) if sujeto else (10, 5, 5, 5), w=w)
        return fn

    # ---------------------------------------------------------------- personajes
    def _max(s, t):
        E = s.E
        if t < T_TERMO:
            return max_objs(E, 'g16_espera_respirando', 1.0 + (t - s.INI), PISO, CAB_YAW, pista=s.pista, t_escena=t, suelo=PISO[2])
        if t < T_CORRE:
            E.max.cara_manual = {'jawOpen': .75} if T_BOCA0 + .25 < t < T_BOCA1 - .12 else {}
            c = clip(E, BASE); tc = .3 + ((t - T_TERMO) * .95) % (c.dur - .6)
            o = max_objs(E, BASE, tc, MAX_P, MAX_YAW, pista=s.pista, t_escena=t, suelo=L.ACERA_Z, ik=s._ik(t))
            E.max.cara_manual = {}
            return o
        p = colocar(E, 'sv_correr_potente', 0.0, MAX_P, -90.0)
        return max_objs(E, 'sv_correr_potente', (t - T_CORRE) * 1.0, p, -90.0, pista=s.pista, t_escena=t, suelo=L.ACERA_Z)

    def _wrap(s, M, t):
        """El shawarma: en el puño izquierdo (desde el primer cuadro en que se ve a Max afuera) hasta que lo deja en el mostrador."""
        E = s.E
        Rl = np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]).T                  # modelo X->z mano, Y->y mano (largo), Z->-x mano
        Rl = np.stack([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]], 1)
        if t < T_TERMO: return []
        if t >= T_DEJA1:                                                       # ya está en el mostrador (no se mueve más)
            return RV.modelo(E.r, 'tp_kebab_wrap', tuple(MOSTRADOR + (0, -.05, 0)), 0.0, largo=LARGO)
        W = mano_mat(E, M, 'Left'); pal = IK.palma_local('Left')
        c = pal + np.array([0, .05, 0])                                        # centro del shawarma: en la palma, un poco hacia el índice
        return [dict(d, resalta=True) for d in objeto_en(E.r, 'tp_kebab_wrap', W, c - Rl @ np.array([0, 0, ZC]), Rl, largo=LARGO)]

    def _vendedor(s, t):
        if t < T_TERMO: return []
        tc = (t * .9) % (s.c_vend.dur - .3) + .1
        o = s.vend.objetos(s.c_vend, tc, tuple(VEND), VEND_GIRO, cara='neutral')
        W = s.vend.hueso_mat('Bip01 R Hand')                                   # cuchillo largo en el puño
        Mk = np.eye(4, dtype=np.float32); Mk[:3, :3] = W[:3, :3] @ np.diag([.26, .004, .035]); Mk[:3, 3] = W[:3, 3] + W[:3, :3] @ np.array([.20, 0, .0])
        Mh = np.eye(4, dtype=np.float32); Mh[:3, :3] = W[:3, :3] @ np.diag([.11, .022, .03]); Mh[:3, 3] = W[:3, 3] + W[:3, :3] @ np.array([.04, 0, .0])
        return o + [dict(mesh='box', model=Mk, texture='white', color=(.82, .84, .86), roughness=.2),
                    dict(mesh='box', model=Mh, texture='white', color=(.12, .1, .09), roughness=.6)]

    # ---------------------------------------------------------------- imagen
    def cuadro(s, E, t):
        o, M, J = s._max(t)
        if t >= T_TERMO: o = s.sud.aplicar(o, min(1.0, max(0.0, (t - 38.0) / 2.6)))
        objs = s.fijos + o + s._wrap(M, t) + s._vendedor(t) + s._termometro() + DB.doner(E, t)
        E.posts = []; gan = 1.0
        if t < T_TERMO:                                                 # desde dentro de la cabina: se abre al calor
            u = (t - s.INI) / (T_TERMO - s.INI)
            ojo = PISO + np.array([-.40, .30, 2.02]); mira = PISO + np.array([2.5, -.45, 1.25])
            cam = (tuple(ojo + (-.05 * u, 0, 0)), tuple(mira), 62)
            gan = 1 + 3.5 * max(0.0, 1 - u * 1.6)
        elif t < T_ORBITA:
            u = (t - T_TERMO) / (T_ORBITA - T_TERMO); p = TERMO + (0, 0, 2.75)
            cam = (tuple(p + (-.35, -1.6 + .25 * u, -.25)), tuple(p), 30)
        elif t < T_MUERDE:
            u = (t - T_ORBITA) / (T_MUERDE - T_ORBITA); h = J['Hips']
            a = math.radians(-105 + 62 * u); R = 2.05
            cam = (tuple(h + np.array([math.cos(a) * R, math.sin(a) * R, .42])), tuple(h + (0.05, 0, .40)), 44)
        elif t < T_DEVUELVE:
            u = (t - T_MUERDE) / (T_DEVUELVE - T_MUERDE); h = J['Head']
            cam = (tuple(h + (.92, -.52 + .08 * u, -.06)), tuple(h + (.05, .04, -.10)), 38 - 3 * u)
        elif t < T_CORRE:                                                # lo deja en el mostrador (los dos en cuadro)
            u = (t - T_DEVUELVE) / (T_CORRE - T_DEVUELVE)
            cam = (tuple(MOSTRADOR + np.array([.62 - .08 * u, -.42, .50])), tuple(MOSTRADOR * .45 + J['Head'] * .55 + (0, 0, -.03)), 52)     # desde el carrito: lo deja en el mostrador con cara de asco
        else:
            u = (t - T_CORRE) / (s.FIN - T_CORRE)
            cam = claves(u, [(0, (16.2, -3.6, 1.6), (14.6, 1.2, 1.1), 52), (1, (15.0, -3.9, 1.7), (12.6, 1.6, 1.2), 52)])
        k = rampa(t, T_BOCA0 + .25, .10, .55, .20)                       # el shawarma resaltado cuando llega a la boca
        img = render_resaltado(E, objs, cam, DB.luz(cam[1], cam[0]), lambda d: d.get('resalta', False), k)
        if gan > 1:
            a = img.astype(np.float32) / 255; a = 1 - (1 - a) ** gan
            img = (a * 255).astype(np.uint8)
        if t < T_ORBITA + .4: img = calor(img, t, .6 if t < T_TERMO else .3)
        return img

    def _termometro(s):
        import tokio as TK
        return TK.reloj(s.E, '46', TERMO, 180.0)


def calor(img, t, k):
    """Aire que tiembla por el calor: ondas verticales finas que mueven las filas de la imagen."""
    H, W = img.shape[:2]; y = np.arange(H)
    dx = (np.sin(y * .045 + t * 9) * 2.2 + np.sin(y * .11 - t * 13) * 1.2) * k * (W / 1080)
    xs = (np.arange(W)[None, :] + dx[:, None]).astype(int) % W
    return img[y[:, None], xs]
