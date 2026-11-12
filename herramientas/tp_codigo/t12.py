"""T12 · La fila de polvo (83,35–89,2 s), en la cocina, la misma cabina del gancho.
83,35 a ras del piso, igual que el gancho: el montoncito del sushi en la cabina ("así que el verdadero tú murió por un sushi")
84,40 travelling lateral largo por la fila de montoncitos hasta la escoba: la imagen premio ("o quién sabes y mucho antes")
86,90 corte seco: los dos Max mirando a cámara; parpadean exactamente a la vez; 88,45 corte de todo (negro)."""
import math, numpy as np
from tp_base import *
import casa_tp as K, casa_lovely as C, montones as MT

T_FILA, T_DOS, T_NEGRO = 84.4, 86.9, 88.45
DOS = [K.PISO + K.F * 1.05 + K.S * .31, K.PISO + K.F * 1.05 - K.S * .31]


class Toma:
    INI, FIN = 83.35, 89.2

    def preparar(s, E):
        C.preparar(E); s.E = E
        s.fijos = K.fijos(E) + [dict(d, resalta=True) for d in MT.objetos(E, tuple(K.PISO), giro=30, escala=.62)]      # el del sushi, en la cabina
        s.pista = caras([(-9, 99, 'neutral')])
        s.W, s.H = E.r.w, E.r.h

    def cuadro(s, E, t):
        E.posts = []
        if t >= T_NEGRO: return np.zeros((s.H, s.W, 3), np.uint8)
        if t < T_FILA:
            u = (t - s.INI) / (T_FILA - s.INI); c = K.PISO
            cam = (tuple(c + K.F * (1.05 - .15 * u) + K.S * .16 + (0, 0, .10)), tuple(c + (0, 0, .07)), 50)
            k = rampa(t, s.INI + .15, .12, .70, .25)                         # el montoncito del sushi, resaltado
            return render_resaltado(E, s.fijos, cam, C.luz(cam[1], cam[0]), lambda d: d.get('resalta', False), k)
        if t < T_DOS:
            u = suave((t - T_FILA) / (T_DOS - T_FILA))
            pts = [np.array([x, y, C.Z0]) for x, y in K.FILA]
            # a lo largo de la pared oeste (bajando en Y) y doblando por la sur hasta la escoba
            ruta = [np.array([13.95, 6.6, .32]), np.array([13.95, 4.4, .30]), np.array([14.4, 4.95, .30]), np.array([16.1, 4.95, .34])]
            miras = [np.array([13.05, 5.9, .10]), np.array([13.05, 4.0, .10]), np.array([14.5, 3.95, .10]), np.array([16.4, 3.75, .45])]
            k = min(int(u * 3), 2); v = u * 3 - k
            ojo = mezcla(ruta[k], ruta[k + 1], v); mira = mezcla(miras[k], miras[k + 1], v)
            cam = (tuple(ojo), tuple(mira), 54)
            return E.render(s.fijos, cam, C.luz(cam[1], cam[0]))
        E.max.cara_manual = {'eyeBlinkLeft': 1.0, 'eyeBlinkRight': 1.0} if 87.95 < t < 88.12 else {}
        a, M1, J1 = max_objs(E, 'g16_espera_respirando', 1.0, DOS[0], K.YAW, pista=s.pista, t_escena=t, suelo=C.Z0)
        b, M2, J2 = max_objs(E, 'g16_espera_respirando', 1.0, DOS[1], K.YAW, pista=s.pista, t_escena=t, suelo=C.Z0, otro=True)
        E.max.cara_manual = {}
        c = (J1['Head'] + J2['Head']) / 2; u = (t - T_DOS) / (T_NEGRO - T_DOS)
        cam = (tuple(c + K.F * (2.9 - .45 * u) + (0, 0, -.10)), tuple(c + (0, 0, -.24 + .06 * u)), 42)
        return E.render(s.fijos + a + b, cam, C.luz(cam[1], cam[0]))
