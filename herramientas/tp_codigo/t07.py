"""T7 · México: el aire flojo (43,7–52,05 s), mañana fresca.
43,70 Max respira hondo con los brazos abiertos, aliviado ("Allá sí hace fresco")
45,30 grúa hacia arriba: la ciudad entre montañas y nubes bajitas ("porque está tan alto que hasta el aire es más flojo")
47,60 bajada en picada hasta Max
48,40 el taquero sirve; Max, verde, empuja el plato despacito; vértigo ("te mareas, se te va el hambre")
50,20 camina torcido hacia la cabina y arranca a correr ("y corres de vuelta a la cabina")."""
import math, os, numpy as np
from tp_base import *
import mexico as MX, actor_rb, ik_tp as IK
import recursos_vel as RV

T_GRUA, T_PICADA, T_PLATO, T_VUELVE = 45.3, 47.6, 48.4, 50.2
MAX_P = np.array([16.95, .02, L.ACERA_Z]); MAX_YAW = 180.0              # frente al carrito (mira +Y)
TAQ = np.array([16.95, 1.92, L.ACERA_Z]); TAQ_GIRO = 0.0
PLATO = np.array([16.95, .46, 1.268]); TACO = .19
COMER = os.environ.get('COMER', 'tp_comer_detenerse_1')        # sube el taco a la boca, se detiene mareado y lo baja


class Toma:
    INI, FIN = 43.7, 52.05

    def preparar(s, E):
        MX.preparar(E); s.E = E
        s.fijos = MX.escena(E, 0.0)
        s.taq = actor_rb.ActorRB(E.r, 'rb_Male_Adult_17', 'taq', 1.74)
        s.c_taq = Clip(ruta_anim('tp_taquero_cortar_servir_1'))
        s.pista = caras([(-9, 45.0, 'contento'), (45.0, 47.6, 'confundido'), (47.6, 50.2, 'asco'), (50.2, 99, 'susto')])
        try: clip(E, COMER); s.comer = COMER
        except FileNotFoundError: s.comer = 'sv_comer'
        s.p_vuelve = colocar(E, 'tp_caminar_mareado_1', 0.0, MAX_P, -90.0)

    def _max(s, t):
        E = s.E
        if t < T_GRUA:
            return max_objs(E, 'tp_estirar_brazos_2', .3 + (t - s.INI) * 1.1, MAX_P + (0, -.35, 0), -15.0, pista=s.pista, t_escena=t, suelo=L.ACERA_Z)
        if t < T_PLATO:
            return max_objs(E, 'g16_espera_respirando', 1 + (t - T_GRUA) * .8, MAX_P, MAX_YAW, pista=s.pista, t_escena=t, suelo=L.ACERA_Z)
        if t < T_VUELVE:
            return max_objs(E, 'g16_espera_respirando', 1.0 + (t - T_PLATO) * .8, MAX_P, MAX_YAW, pista=s.pista, t_escena=t, suelo=L.ACERA_Z, ik=s._ik(t))
        if t < 51.4:
            return max_objs(E, 'tp_caminar_mareado_1', (t - T_VUELVE) * 1.0, s.p_vuelve, -90.0, pista=s.pista, t_escena=t, suelo=L.ACERA_Z)
        o, M, J = max_objs(E, 'tp_caminar_mareado_1', 1.2, s.p_vuelve, -90.0, suelo=L.ACERA_Z)
        p = colocar(E, 'sv_correr_potente', .35, J['Hips'], -90.0)
        return max_objs(E, 'sv_correr_potente', .35 + (t - 51.4), p, -90.0, pista=s.pista, t_escena=t, suelo=L.ACERA_Z)

    def _verde(s, o, g):
        """Max verde (mareo): la piel toma un tono verdoso."""
        if g <= 0: return o
        piel = {f'max_m{i}' for i, q in enumerate(s.E.max.prims) if q['mname'].startswith('Std_Skin')}
        for d in o:
            if d['mesh'] in piel: d['color'] = (1 - .22 * g, 1.0, 1 - .30 * g)
        return o

    def _ik(s, t):
        """La mano izquierda lleva el taco con la palma hacia arriba y un extremo hacia la boca (la muñeca la mueve el clip)."""
        def fn(m, G, Pm, b):
            U, Fw, Rt = IK.ejes(m, G, b, 'Chest'); pc = Pm[m.idx['Chest']]; pal = IK.palma_local('Left', .085, .014)
            Uh, Fh, Rh = IK.ejes_cabeza(m, G, b); Pb, nb = IK.punto_cara(m, G, Pm, b, 'boca')
            Fa = IK.marco_de(Fw * .85 - U * .35 + Rt * .15, U * .85 - Fw * .35)             # palma arriba, un extremo hacia él
            Fb = IK.marco_de(Fh * .80 - Uh * .50 + Rh * .15, Uh * .70 - Fh * .55)
            Wa = pc + Fw * .27 - Rt * .13 - U * .06
            Wb = Pb + nb * .085 - Uh * .035 - Fb @ (pal + np.array([0, -TACO / 2, .03]))       # el extremo del taco a 8 cm de la boca: ahí se frena
            Wc = pc + Fw * .40 - Rt * .22 - U * .20                                           # lo aparta
            ub = suave((t - T_PLATO - .05) / .55) * (1 - suave((t - 49.05) / .55)); uc = suave((t - 49.45) / .55)
            W = Wa * (1 - ub) + Wb * ub; F = IK._mez(Fa, Fb, ub)
            W = W * (1 - uc) + Wc * uc
            s.falta = IK.brazo(m, G, Pm, b, 'Left', W, F, polo=-U * .8 - Rt * .5 - Fw * .2, w=1.0)
            IK.dedos(m, G, Pm, b, 'Left', {d: (30, 34, 22) for d in IK.DEDOS}, pulgar=(30, 8, 15, 15))
            IK.brazo_de(m, G, Pm, b, 'Right', clip(s.E, 'g16_espera_respirando'), 1.0)
        return fn

    def _taco(s, M, t):
        """El taco en la mano izquierda de Max mientras le da el mareo (desde el primer cuadro de ese plano); el plato en el mostrador."""
        E = s.E; o = []
        Mp = np.eye(4, dtype=np.float32); Mp[:3, :3] = np.diag([.26, .26, .016]); Mp[:3, 3] = PLATO
        o.append(dict(mesh='box', model=Mp, texture='white', color=(.95, .95, .93), roughness=.25))
        en_mano = T_PLATO <= t < T_VUELVE
        for k, dx in enumerate((-.075, 0, .075)):
            if k == 1 and en_mano: continue
            o += RV.modelo(E.r, 'tp_taco', tuple(PLATO + (dx, 0, .008)), 90 + 5 * k, largo=TACO)
        if en_mano:
            W = mano_mat(E, M, 'Left'); pal = IK.palma_local('Left', .085, .014)
            Rl = np.stack([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]], 1)              # modelo X (largo) -> y mano; Z (abertura) -> fuera de la palma
            o += objeto_en(E.r, 'tp_taco', W, pal, Rl, largo=TACO)
        return o

    def cuadro(s, E, t):
        o, M, J = s._max(t)
        g = 0.0 if t < 47.9 else min(1.0, (t - 47.9) / 1.2)
        o = s._verde(o, g)
        tt = (t * .8) % (s.c_taq.dur - .4) + .2
        objs = s.fijos + o + s.taq.objetos(s.c_taq, tt, tuple(TAQ), TAQ_GIRO, cara='neutral') + MX.delantal(s.taq)
        objs += s._taco(M, t)
        E.posts = []; post = None
        if t < T_GRUA:
            u = (t - s.INI) / (T_GRUA - s.INI); h = J['Head']
            cam = (tuple(h + (.55 - .15 * u, -1.75, -.30)), tuple(h + (0, 0, -.30)), 48)
        elif t < T_PICADA:
            u = suave((t - T_GRUA) / (T_PICADA - T_GRUA)); h = J['Hips']
            ojo = mezcla(h + (-1.0, -2.4, .4), h + (-1.5, -9.0, 15.0), u)
            mira = mezcla(h + (0, 0, .5), h + (0, 30, 9), u)
            cam = (tuple(ojo), tuple(mira), 50 + 10 * u)
        elif t < T_PLATO:
            u = ((t - T_PICADA) / (T_PLATO - T_PICADA)) ** 2; h = J['Head']
            ojo = mezcla(J['Hips'] + (-1.5, -9.0, 15.0), h + np.array([.30, 1.0, .02]) / 1.044 * 1.25, u)
            mira = mezcla(J['Hips'] + (0, 30, 9), h + (0, .15, -.25), u)
            cam = (tuple(ojo), tuple(mira), 60 - 16 * u)
        elif t < T_VUELVE:
            u = (t - T_PLATO) / (T_VUELVE - T_PLATO); h = J['Head']
            d = 1.25 - .55 * suave(u); fov = 2 * math.degrees(math.atan(math.tan(math.radians(18)) * 1.25 / d))   # dolly zoom
            vai = .06 * math.sin(t * 5.0)
            dc = np.array([float(x) for x in os.environ.get('CAMD', '-.55,.85,.25').split(',')]); dc /= np.linalg.norm(dc)
            ojo = h + dc * d + (0, 0, vai)      # tres cuartos desde su izquierda: se ven la cara verde y el taco, sin el carrito por delante
            cam = (tuple(ojo), tuple(h + (0, .05, -.17 + vai * .5)), fov)
        else:
            u = (t - T_VUELVE) / (s.FIN - T_VUELVE)
            cam = claves(u, [(0, (17.6, -3.4, 1.55), (16.2, .8, 1.1), 52), (1, (16.6, -3.6, 1.6), (13.2, 1.6, 1.2), 52)])
        return E.render(objs, cam, MX.luz(cam[1], cam[0]))
