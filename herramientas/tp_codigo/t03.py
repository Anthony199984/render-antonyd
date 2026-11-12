"""T3 · El escáner por dentro (12,3–17,5 s).
12,3  Max en la cabina, contento y con los ojos abiertos (cerrados se veían mal de cerca); un anillo de luz verde lo recorre; dolly zoom hacia la frente
13,25 fundido rosado: entramos al paisaje por dentro; la línea verde barre las células ("cada célula")
14,75 la línea pasa por las fotos-recuerdo (el perro, la esposa, el sushi) y las lee ("cada recuerdo")
15,8  salida rápida hacia atrás: Max con cara de spa y los pies empiezan a deshacerse ("y después te desintegra")."""
import math, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, cabina as CB, arena as AR, dentro as DT
import real_scene as RS, skin

T_DENTRO, T_FOTOS, T_SALE = 13.25, 14.75, 15.8
CIERRA = float(__import__('os').environ.get('CIERRA', 0))   # párpados: con 1.0 se atraviesan y de cerca se ve una raya oscura
BAJA = float(__import__('os').environ.get('BAJA', .3))            # la cámara baja un poco: Max tiene la cabeza inclinada y así se le ve la cara de frente


def _anillo(r, nombre, R=.47, grosor=.018, n=96):
    a = np.linspace(0, 2 * math.pi, n + 1); P = []; N = []
    for i in range(n):
        for (x, z) in [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]:
            ang = a[i + x]; P.append([R * math.cos(ang), R * math.sin(ang), z * grosor]); N.append([math.cos(ang), math.sin(ang), 0])
    P = np.array(P, np.float32); N = np.array(N, np.float32)
    d = skin.interleave(P, N, np.zeros((len(P), 2), np.float32)); d[:, 8:11] = 1
    r.add_mesh(nombre, d)


class Toma:
    INI, FIN = 12.3, 17.5

    def preparar(s, E):
        C.preparar(E); s.E = E
        s.fijos = K.fijos(E)
        s.pista = caras([(-9, 99, __import__('os').environ.get('CARA3', 'contento'))])
        _anillo(E.r, 'escaner_anillo')
        s.D = DT.Paisaje(E, semilla=3)
        for n, c, g in [('perro', (-1.9, 8.6, 1.35), 14), ('esposa', (.15, 9.6, 1.55), -4), ('sushi', (2.0, 8.9, 1.25), -16)]:
            s.D.agregar_foto(n, f'{RAIZ}/fotos/{n}.jpg', c, g, ancho=1.55)
        s.A = AR.Arena(E, K.PISO, t_ini=16.45, t_fin=16.45 + 9.0, z_ini=0.0, z_fin=2.1)
        s.t_pre = T_SALE

    def _max(s, t):
        k = CIERRA * (1 - suave((t - T_SALE - .05) / .35))                # al salir del viaje abre los ojos
        s.E.max.cara_manual = {'eyeBlinkLeft': k, 'eyeBlinkRight': k}
        o, M, J = max_objs(s.E, 'g16_espera_respirando', 1.0 + (t - s.INI) * .8, K.PISO, K.YAW, pista=s.pista, t_escena=t, suelo=K.PISO[2])
        return o, M, J

    def avanzar(s, t):
        if t < T_SALE: return
        o, M, J = s._max(t); s.A.avanzar(t); s._o = (s.A.cortar(s.E, o, M), M, J)

    # ---------------------------------------------------------------- afuera
    def _afuera(s, E, t):
        if t >= T_SALE: o, M, J = s._o
        else: o, M, J = s._max(t)
        objs = s.fijos + o
        if t < T_DENTRO + .1:                                          # anillo verde que baja de la cabeza a los pies
            z = 2.0 - 2.0 * min(max((t - 12.35) / .85, 0), 1)
            Ma = np.eye(4, dtype=np.float32); Ma[:3, 3] = K.PISO + (0, 0, z)
            objs.append(dict(mesh='escaner_anillo', model=Ma, texture='white', color=(.4, 1, .55), emission=8, noshadow=True))
        if t >= T_SALE: objs += s.A.monton(E); E.posts = [s.A.post]
        else: E.posts = []
        h = J['Head']; fr = h + K.F * .10 + (0, 0, .07)                   # la frente
        if t < T_DENTRO:
            u = suave((t - s.INI) / (T_DENTRO - s.INI))
            d = .95 - .57 * u; fov = 26 + 20 * u                           # dolly zoom suave: sin deformar la cara
            cam = (tuple(fr + K.F * d + K.S * .05 - (0, 0, .42 * d * BAJA)), tuple(fr + (0, 0, .05 * u)), fov)
        else:
            u = suave((t - T_SALE) / .6)
            ojo0 = fr + K.F * .85 - (0, 0, .42 * .85 * BAJA); ojo1 = K.PISO + K.F * 2.3 - K.S * .35 + (0, 0, 1.15)
            mira = mezcla(fr, K.PISO + (0, 0, .75), u)
            cam = (tuple(mezcla(ojo0, ojo1, u)), tuple(mira), 30 + 22 * u)
            if t > T_SALE + .6:                                            # baja despacio hacia los pies que se deshacen
                v = suave((t - T_SALE - .6) / 1.1)
                cam = (tuple(ojo1 + (0, 0, -.55 * v) - K.F * .4 * v), tuple(mezcla(K.PISO + (0, 0, .75), K.PISO + (0, 0, .35), v)), 52)
        E.grado = {}
        return E.render(objs, cam, C.luz(cam[1], cam[0]))

    # ---------------------------------------------------------------- adentro
    def _adentro(s, E, t):
        u = (t - T_DENTRO) / (T_SALE - T_DENTRO)
        ly = 1.0 + 10.5 * u                                                # la línea avanza y cruza las fotos al final
        y = -1.5 + 6.0 * suave(u)                                          # la cámara la sigue a ras de las células
        ojo = np.array([.25 * math.sin(t * 1.2), y, .95 + .10 * math.sin(t * 2)])
        mira = np.array([0, y + 5, .45])
        foco = mezcla(ojo, mira, .45)
        if t > T_FOTOS - .3:
            v = suave((t - T_FOTOS + .3) / (T_SALE - T_FOTOS + .3))
            ojo = ojo + (0, 0, .45 * v); mira = mezcla(mira, (0, 9.0, 1.4), v); foco = mezcla(foco, (0, 9.0, 1.4), v)
        cam = (tuple(ojo), tuple(mira), 56)
        lect = {}
        for f in s.D.fotos_cfg:
            l = (ly - f['c'][1] + .6) / 1.0
            if l > 0: lect[f['n']] = min(l, 1.0)
        objs = s.D.objetos(t, linea_y=ly) + s.D.fotos(t, lect)
        E.posts = []
        E.grado = dict(foco=tuple(foco), cm=cam, k_dof=1.2)
        img = E.render(objs, cam, s.D.luz(cam, B))
        E.grado = {}
        return img

    def cuadro(s, E, t):
        if t < T_DENTRO - .12: return s._afuera(E, t)
        if t < T_DENTRO + .12:                                              # fundido rosado de la frente al paisaje
            u = (t - (T_DENTRO - .12)) / .24
            a = s._afuera(E, t) if u < .5 else None; b = s._adentro(E, t) if u >= .5 else None
            rosa = np.array([235, 150, 160], np.float32)
            base = a if a is not None else b; k = 1 - abs(u - .5) * 2
            return np.clip(base * (1 - k * .85) + rosa * k * .85, 0, 255).astype(np.uint8)
        if t < T_SALE: return s._adentro(E, t)
        return s._afuera(E, t)
