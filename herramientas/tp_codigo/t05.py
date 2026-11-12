"""T5 · Tokio cerró (27,95–36,3 s).
27,95 desde atrás a ras del piso: Max camina feliz hacia el restaurante ("solo que con la diferencia de horario")
30,40 la cortina baja de golpe a centímetros de su nariz cuando el reloj pasa de 23:59 a 00:00; empuje a la cara ("medianoche")
31,60 inserto del reloj de la calle: 00:00
32,15 mano en la panza, le ruge ("y como el restaurante ya cerró")
33,20 plano general: se da la vuelta y vuelve a la cabina arrastrando los pies ("buscas un sitio abierto, un shawarma en Dubai")."""
import math, numpy as np
from tp_base import *
import tokio as TK

T_CORTINA, T_RELOJ, T_PANZA, T_VUELVE = 30.40, 31.60, 32.15, 33.20
T_CARA5 = 30.78                                     # el plano de atrás sigue hasta que la cortina golpea el piso; después, su cara
T_GOLPE = 30.62                                     # la cortina toca el piso (y el reloj cambia a 00:00)
CORT = TK.RESTAURANTE + np.array([0, .05, 0])       # base de la cortina (mirando a +Y)
PARADO = np.array([.25, CORT[1] + .40, TK.ACERA])   # la cadera de Max frente a la cortina
RELOJ_POS = np.array([1.80, -TK.ANCHO - .22, TK.ACERA]); RELOJ_GIRO = 0.0


def _suelo(y):
    """La acera sur sube 12 cm: rampa corta para que los pies no salten."""
    return TK.ACERA * min(max((-y - (TK.ANCHO - .15)) / .3, 0.0), 1.0)


class Toma:
    INI, FIN = 27.95, 36.3

    def preparar(s, E):
        TK.preparar(E); s.E = E
        s.fijos = TK.escena(E, 0.0)
        s.pista = caras([(-9, T_GOLPE, 'contento'), (T_GOLPE, 31.6, 'shock'), (31.6, 33.3, 'triste'), (33.3, 99, 'cansado')])
        s.p_camina = colocar(E, 'g15_caminar', 3.05, PARADO, 0.0)
        s.p_frena = colocar(E, 'sv_frenar_barriga', 2.2, PARADO, 0.0)
        s.yaw_vuelve = -52.0
        s.p_vuelve = colocar(E, 'tp_caminar_triste_2', 0.0, PARADO, s.yaw_vuelve)

    def _max(s, t):
        E = s.E
        if t < T_CORTINA:
            tc = .6 + (t - s.INI) * (2.45 / (T_CORTINA - s.INI))
            o, M, J = max_objs(E, 'g15_caminar', tc, s.p_camina, 0.0, pista=s.pista, t_escena=t, suelo=0.0)
            sz = _suelo(J['Hips'][1])
            if sz > 0: o, M, J = max_objs(E, 'g15_caminar', tc, s.p_camina, 0.0, pista=s.pista, t_escena=t, suelo=sz)
            return o, M, J
        if t < T_VUELVE:
            tc = 1.85 + (t - T_CORTINA) * .62
            return max_objs(E, 'sv_frenar_barriga', tc, s.p_frena, 0.0, pista=s.pista, t_escena=t, suelo=TK.ACERA)
        tc = (t - T_VUELVE) * 1.05
        o, M, J = max_objs(E, 'tp_caminar_triste_2', tc, s.p_vuelve, s.yaw_vuelve, pista=s.pista, t_escena=t, suelo=TK.ACERA)
        sz = _suelo(J['Hips'][1])
        if sz < TK.ACERA: o, M, J = max_objs(E, 'tp_caminar_triste_2', tc, s.p_vuelve, s.yaw_vuelve, pista=s.pista, t_escena=t, suelo=sz)
        return o, M, J

    def cuadro(s, E, t):
        if t >= T_VUELVE and not hasattr(s, 'Fv'):                           # hacia dónde camina de verdad (medido en el clip, ANTES de posar el cuadro)
            pa = s._max(T_VUELVE + .3)[2]['Hips']; pb = s._max(s.FIN - .2)[2]['Hips']; d = (pb - pa) * (1, 1, 0); s.Fv = d / np.linalg.norm(d)
        o, M, J = s._max(t)
        u_c = 0.0 if t < T_CORTINA + .05 else min(1.0, ((t - T_CORTINA - .05) / (T_GOLPE - T_CORTINA - .05)) ** 2)
        hora = '00:00' if t >= T_GOLPE else '23:59'
        objs = s.fijos + o + TK.cortina(E, CORT, u=u_c, pintada=True) + [dict(d, resalta=True) for d in TK.reloj(E, hora, RELOJ_POS, RELOJ_GIRO)]
        E.posts = []
        if t < T_CARA5:
            u = (t - s.INI) / (T_CARA5 - s.INI); h = J['Hips']
            cam = (tuple(h + (.75, 1.75, -.55)), tuple(h + (-.25, -2.0, .40)), 58)       # de atrás y de lado: se ven él, la cortina y el reloj
            foco, k = h + (0, 0, .4), 1.9                                # fondo bien desenfocado (los letreros no se leen)
        elif t < T_RELOJ:
            u = suave((t - T_CARA5) / (T_RELOJ - T_CARA5)); h = J['Head']
            ojo = mezcla(h + (1.30, -.30, -.12), h + (.62, -.22, -.04), u)        # de perfil: la cara y la cortina detrás
            cam = (tuple(ojo), tuple(h + (0, -.12, -.05)), 46)
            foco, k = h, 1.6
        elif t < T_PANZA:
            u = (t - T_RELOJ) / (T_PANZA - T_RELOJ); rp = RELOJ_POS + (0, 0, 2.75)
            cam = (tuple(rp + (.35, 3.6 - .35 * u, -.45)), tuple(rp), 30)             # más lejos: el 00:00 entra COMPLETO en el cuadro
            foco, k = rp, 2.0
        elif t < T_VUELVE:
            u = (t - T_PANZA) / (T_VUELVE - T_PANZA); h = J['Hips']
            cam = (tuple(h + (1.15, -.10, .42 + .05 * u)), tuple(h + (0, -.15, .32)), 50)
            foco, k = h + (0, 0, .4), 1.4
        else:
            u = (t - T_VUELVE) / (s.FIN - T_VUELVE); h = J['Hips']
            Fv = s.Fv; Sv = np.array([-Fv[1], Fv[0], 0])
            hs = np.array([h[0], h[1], TK.ACERA + .95])                        # lo acompaña de frente y de lado, sin el rebote de la cadera
            ojo = hs + Fv * (2.5 - .5 * u) + Sv * (1.0 + .5 * u) + (0, 0, .45)
            cam = (tuple(ojo), tuple(hs + (0, 0, .38)), 44)
            foco, k = h, .8
        E.grado = dict(sat=1.15, con=1.10, foco=tuple(foco), cm=cam, k_dof=k)
        k = rampa(t, T_RELOJ + .02, .08, .38, .12)                      # el 00:00 resaltado
        return render_resaltado(E, objs, cam, TK.luz(cam[1], cam[0]), lambda d: d.get('resalta', False), k)
