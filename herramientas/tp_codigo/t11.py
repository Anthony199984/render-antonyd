"""T11 · Lo que quedó en cada cabina (73,36–83,35 s): paisaje por dentro.
73,36 (el montaje entra por los ojos desde la T10) — pasillo de pantallas-recuerdo flotando sobre las células: en cada
      cabina un montoncito de polvo: casa, Tokio, Dubái, México ("Y ahí lo entiendes todo, ninguno de los dos es el original")
77,40 se detiene en Tokio: el barrendero lo echa al recogedor sin mirarlo ("porque para teletransportarte")
79,40 afuera: los dos Max con la misma cara de horror y la mano en la boca a la vez ("la máquina tiene que matarte")
80,90 vuelta al pasillo, que se aleja ("y armar una copia exactamente igual a ti")."""
import math, glob, numpy as np
from tp_base import *
import dentro as DT, pantalla as PT
from PIL import Image

T_TOKIO, T_AFUERA, T_VUELTA = 77.4, 79.4, 80.9
PANT = {'casa': (-1.35, 6.0, 1.45, 14), 'tokio': (1.35, 8.6, 1.50, -14), 'dubai': (-1.35, 11.2, 1.45, 14), 'mexico': (1.35, 13.8, 1.50, -14)}


class Toma:
    INI, FIN = 73.36, 83.35

    def preparar(s, E):
        s.E = E
        s.D = DT.Paisaje(E, semilla=7, tejido=np.array([.80, .40, .46]))
        s.tok = sorted(glob.glob(f'{RAIZ}/fotos/tokio_barre/*.jpg'))
        for n, (x, y, z, g) in PANT.items():
            im = f'{RAIZ}/fotos/pant_{n}.jpg' if n != 'tokio' else s.tok[0]
            s.D.agregar_foto(n, im, (x, y, z), g, ancho=1.75)
        s.cuarto = None

    def _tokio(s, t):
        u = min(1.0, max(0.0, (t - T_TOKIO - .1) / 1.8))
        k = int(u * (len(s.tok) - 1))
        if k != getattr(s, '_k', -1):
            im = Image.open(s.tok[k]).convert('RGB'); w, h = im.size; b = int(w * .06)
            marco = Image.new('RGB', (w + 2 * b, h + b * 4), (245, 242, 236)); marco.paste(im, (b, b))
            PT.poner(s.E.r, 'foto_tokio', marco.convert('RGBA')); s._k = k

    def _adentro(s, E, t):
        s._tokio(t)
        def parada(n, d, dz=0.0):                                       # delante de una foto, a d metros, mirándola de frente
            x, y, z, g = PANT[n]; fr = np.array([x, y, z]); nn = np.array([math.sin(math.radians(g)), -math.cos(math.radians(g)), 0])
            return fr + nn * d + (0, 0, .05 + dz), fr
        if t < T_AFUERA:                                                # la foto de casa y después la de Tokio, cada una llenando el cuadro
            K = [(s.INI, (np.array([0.0, 1.2, 1.40]), np.array([-.5, 6.0, 1.35]))), (74.45, parada('casa', 2.15)), (75.75, parada('casa', 1.85)),
                 (77.05, parada('tokio', 2.3)), (T_AFUERA, parada('tokio', 1.85))]
        else:                                                           # de vuelta: Dubái y México, con su montoncito
            K = [(T_VUELTA, parada('dubai', 2.2)), (81.9, parada('dubai', 1.9)), (82.75, parada('mexico', 2.2)), (s.FIN, parada('mexico', 1.9))]
        k = max(0, min(len(K) - 2, sum(1 for q in K if q[0] <= t) - 1))
        (ta, (oa, ma)), (tb, (ob, mb)) = K[k], K[k + 1]; v = suave((t - ta) / (tb - ta))
        ojo = mezcla(oa, ob, v) + (0, 0, .03 * math.sin(t * 1.3)); mira = mezcla(ma, mb, v); foco = mira
        cam = (tuple(ojo), tuple(mira), 54)
        objs = s.D.objetos(t) + s.D.fotos(t, {})
        E.posts = []; E.grado = dict(foco=tuple(foco), cm=cam, k_dof=1.1)
        img = E.render(objs, cam, s.D.luz(cam, B)); E.grado = {}
        return img

    # ---------------------------------------------------------------- afuera: los dos Max en el cuarto
    def _afuera(s, E, t):
        import casa_lovely as C
        if s.cuarto is None:
            C.preparar(E); s.cuarto = C.escena(E, 0.0, cuarto=True, cocina=True)
            s.pista = caras([(-9, 99, 'shock')])
        tc = .35 + (t - T_AFUERA) * 1.3
        a, M1, J1 = max_objs(E, 'tp_tapar_boca_1', tc, (22.62, 6.55, C.Z0), 75.0, pista=s.pista, t_escena=t, suelo=C.Z0)
        b, M2, J2 = max_objs(E, 'tp_tapar_boca_1', tc, (22.80, 5.80, C.Z0), 75.0, pista=s.pista, t_escena=t, suelo=C.Z0, otro=True)
        c = (J1['Head'] + J2['Head']) / 2
        F = np.array([math.sin(math.radians(75)), -math.cos(math.radians(75)), 0])
        u = (t - T_AFUERA) / (T_VUELTA - T_AFUERA)
        cam = (tuple(c + F * (2.75 - .2 * u) + (0, 0, -.15)), tuple(c + (0, 0, -.28)), 46)
        Lz = C.luz(cam[1], cam[0])
        Lz.update(sun_col=(.02, .025, .04), env_top=(.05, .06, .10), env_hor=(.05, .05, .08), env_bot=(.03, .03, .04), env_k=1.0,
                  bg_top=(.01, .015, .04), bg_hor=(.02, .03, .07), bg_bot=(.01, .01, .02))
        Lz['cielo'] = False
        Lz['lights'] = [dict(pos=(26.45, 7.6, 1.1), col=(2.6, 1.9, 1.1)), dict(pos=(26.45, 4.9, 1.1), col=(2.6, 1.9, 1.1)), dict(pos=tuple(np.array(cam[0]) + (0, 0, .4)), col=(2.0, 1.85, 1.7))]            # bien iluminados: se tienen que ver las dos caras iguales
        E.posts = []
        return E.render(s.cuarto + a + b, cam, Lz)

    def cuadro(s, E, t):
        if T_AFUERA <= t < T_VUELTA: return s._afuera(E, t)
        img = s._adentro(E, t)
        if t < s.INI + .45:                              # entra por los ojos: destello rosado que se abre
            k = 1 - (t - s.INI) / .45
            img = np.clip(img * (1 - k * .9) + np.array([240, 160, 170]) * k * .9, 0, 255).astype(np.uint8)
        return img
