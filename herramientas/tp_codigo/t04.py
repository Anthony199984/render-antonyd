"""T4 · Armado en Tokio (17,5–27,95 s).
17,5  afuera de la casa turquesa: un chorro de luz sale del techo y sube al cielo ("Y es que para teletransportarte")
19,0  el globo de noche: la cámara sigue el chorro hasta Japón ("la máquina necesita desarmarte")
21,0  plano bajo en la cabina de Tokio: el chorro baja y Max se arma capa por capa, pies y piernas ("y armarte al otro lado")
23,0  sigue el armado: la camiseta y al final la cara, que abre los ojos sonriendo ("como recuerdas haber entrado")
24,6  plano medio afuera de la cabina: estira los brazos ("sales seguro de que fue un simple viaje")
26,3  infla el pecho y huele el aire, satisfecho ("por un sushi")."""
import math, numpy as np
from tp_base import *
import tokio as TK, cabina as CB, arena as AR, globo as GL
import lovely_vida as V

T_GLOBO, T_TOKIO, T_CARA, T_SALE, T_PECHO = 19.0, 21.0, 23.0, 24.6, 26.3
TECHO = np.array([14.05, 7.25, 6.6])
CAB = TK.CABINA; YAW = TK.CAB_YAW
PISO = CAB + [0, 0, CB.PISO_Z]
F = np.array([math.sin(math.radians(YAW)), -math.cos(math.radians(YAW)), 0.0]); S = np.array([-F[1], F[0], 0.0])
FUERA = PISO - [0, 0, CB.PISO_Z] + F * 1.05          # Max afuera de la cabina (en la acera)


def _haz(r):
    if 'haz_luz' in r.meshes: return
    Mt = V.Malla(); Mt.cil((0, 0, 0), (0, 0, 1), 1.0, (1, 1, 1), 24); Mt.subir(r, 'haz_luz')


def haz(base, alto, radio, brillo, col=(.6, .95, 1.0)):
    M = np.eye(4, dtype=np.float32); M[:3, :3] = np.diag([radio, radio, alto]); M[:3, 3] = base
    Mg = M.copy(); Mg[:3, :3] = np.diag([radio * 2.6, radio * 2.6, alto])
    return [dict(mesh='haz_luz', model=M, texture='white', color=col, emission=9 * brillo, noshadow=True, roughness=1),
            dict(mesh='haz_luz', model=Mg, texture='white', color=tuple(col) + (.18 * brillo,), emission=3 * brillo, transparent=True, noshadow=True, roughness=1)]


class Toma:
    INI, FIN = 17.5, 27.95

    def preparar(s, E):
        s.E = E; s.W, s.H = E.r.w if hasattr(E.r, 'w') else 0, 0
        import os
        s.parte = os.environ.get('PARTE_T4')               # la casa (Lovely) y Tokio no se pueden preparar en el mismo proceso
        s.modo = None

    def _modo(s, E, t):
        m = 'casa' if t < T_GLOBO else ('globo' if t < T_TOKIO else 'tokio')
        if m == s.modo: return
        if m == 'casa':
            import casa_lovely as C; C.preparar(E); s.fijos = C.escena(E, 0.0, cuarto=True, cocina=True)
        elif m == 'tokio':
            TK.preparar(E); s.fijos = TK.escena(E, 0.0)
            s.pista = caras([(-9, 24.25, 'neutral'), (24.25, 25.6, 'contento'), (25.6, 99, 'de_acuerdo')])
            s.A = AR.Arena(E, PISO, t_ini=T_TOKIO + .25, t_fin=24.2, z_ini=0.0, z_fin=2.10, invertir=True)
        _haz(E.r); s.modo = m

    # ---------------------------------------------------------------- partes
    def _casa(s, E, t):
        import casa_lovely as C
        u = (t - s.INI) / (T_GLOBO - s.INI)
        alto = 60 * suave(min(u * 1.6, 1.0)); b = min(1.0, u * 4)
        objs = s.fijos + (haz(TECHO, alto, .22, b) if alto > .05 else [])
        cam = claves(u, [(0, (14.6, -6.5, 1.5), (14.2, 6.0, 6.0), 58), (1, (14.9, -6.0, 1.3), (14.1, 6.5, 12.0), 62)])
        E.posts = []
        return E.render(objs, cam, L.luz(cam[1]))

    def _globo(s, E, t):
        u = (t - T_GLOBO) / (T_TOKIO - T_GLOBO); a = suave(u)
        p, q = GL._xyz(*GL.CASA), GL._xyz(*GL.TOKIO); w = math.acos(np.clip(p @ q, -1, 1))
        v = lambda k: (math.sin((1 - k) * w) * p + math.sin(k * w) * q) / math.sin(w)
        c = v(min(a * 1.05, 1.0)); c = c / np.linalg.norm(c)
        nrm = np.cross(p, q); nrm /= np.linalg.norm(nrm)
        ojo = c * (2.6 - .4 * a) + nrm * (.95 - .3 * a); mira = c * .45
        Lz = B.luces((0, 0, 0)); Lz['cielo'] = False
        Lz.update(bg_top=(.0, .01, .03), bg_hor=(.01, .03, .08), bg_bot=(0, 0, .01), env_k=.05, sun_col=(.08, .1, .15), lights=[])
        E.posts = []
        img = E.render(GL.objetos(E, min(a * 1.08, 1.0)), (tuple(ojo), tuple(mira), 42), Lz)
        if u < .12: img = desenfoque_v(img, int(30 * (1 - u / .12)))      # entra con barrido desde el chorro que sube
        return img

    def _max(s, t):
        E = s.E
        if t < T_SALE:
            if t < 24.25: E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0}
            else: E.max.cara_manual = {}
            return max_objs(E, 'g16_espera_respirando', 1.0 + (t - T_TOKIO) * .6, PISO, YAW, pista=s.pista, t_escena=t, suelo=PISO[2])
        E.max.cara_manual = {}
        if t < T_PECHO:
            return max_objs(E, 'tp_estirar_brazos_1', .2 + (t - T_SALE) * 1.15, FUERA, YAW, pista=s.pista, t_escena=t, suelo=TK.ACERA)
        return max_objs(E, 'tp_respirar_hondo_1', .3 + (t - T_PECHO) * 1.0, FUERA, YAW, pista=s.pista, t_escena=t, suelo=TK.ACERA)

    def avanzar(s, t):
        if t < T_TOKIO: return
        s._modo(s.E, t)
        o, M, J = s._max(t)
        if t < T_SALE:
            s.A.avanzar(t); o = s.A.cortar(s.E, o, M)
        s._o = (o, M, J)

    def _tokio(s, E, t):
        o, M, J = s._o
        objs = s.fijos + o
        if t < T_TOKIO + .9:                                            # el chorro llega del cielo a la cabina
            b = 1 - max(0.0, (t - T_TOKIO - .4) / .5)
            objs += haz(PISO + (0, 0, .05), 40, .20, b)
        E.posts = [s.A.post] if t < T_SALE else []
        if t < T_CARA:
            u = (t - T_TOKIO) / (T_CARA - T_TOKIO)
            cam = claves(u, [(0, PISO + F * 1.6 + S * .3 + (0, 0, .25), PISO + (0, 0, .35), 50), (1, PISO + F * 1.5 + S * .45 + (0, 0, .55), PISO + (0, 0, .85), 50)])
        elif t < T_SALE:
            u = (t - T_CARA) / (T_SALE - T_CARA)
            h = PISO + (0, 0, 1.72)
            cam = claves(u, [(0, PISO + F * 1.5 + S * .4 + (0, 0, 1.05), PISO + (0, 0, 1.25), 46), (.7, h + F * .85 + S * .2 + (0, 0, -.05), h - (0, 0, .03), 38),
                             (1, h + F * .75 + S * .18, h - (0, 0, .03), 36)])
        elif t < T_PECHO:
            u = (t - T_SALE) / (T_PECHO - T_SALE); hp = J['Hips']
            cam = (tuple(hp + F * 2.7 - S * .9 + (0, 0, .45 + .1 * u)), tuple(hp + (0, 0, .55)), 50)
        else:
            u = (t - T_PECHO) / (s.FIN - T_PECHO); h = J['Head']
            cam = (tuple(h + F * 1.3 + S * .55 + (0, 0, -.20 - .05 * u)), tuple(h + (0, 0, -.22)), 40 - 4 * u)
        return E.render(objs, cam, TK.luz(cam[1], cam[0]))

    def cuadro(s, E, t):
        s._modo(E, t)
        if t < T_GLOBO: return s._casa(E, t)
        if t < T_TOKIO: return s._globo(E, t)
        return s._tokio(E, t)
