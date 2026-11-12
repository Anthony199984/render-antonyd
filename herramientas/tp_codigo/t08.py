"""T8 · Se va la luz (52,05–61,3 s), en el local de cabinas de México.
52,05 sobre el hombro: con el celular en la izquierda, toca el ícono de la casita ("Y de regreso te das cuenta")
54,00 la línea verde lo recorre de abajo arriba y la barra se llena: pitido de enviado ("el peor error de tu vida")
56,30 plano general del local ("Viajar a Latinoamérica, porque en pleno viaje")
59,00 los tubos se apagan uno por uno hasta la cabina ("se va la luz")
59,90 negro y silencio total
60,45 primer plano: solo su cara alumbrada por el celular, mirando a cámara."""
import math, os, numpy as np
from tp_base import *
import local_mx as LM, cabina as CB, pantalla as PT, ik_tp as IK
import real_scene as RS, skin

T_ESCANEO, T_GENERAL, T_APAGA, T_NEGRO, T_CARA = 54.0, 56.3, 58.35, 59.9, 60.45
PISO = LM.CABINA + [0, 0, CB.PISO_Z]
F = np.array([math.sin(math.radians(LM.CAB_YAW)), -math.cos(math.radians(LM.CAB_YAW)), 0.0]); S = np.array([-F[1], F[0], 0.0])


def luces_en(t):
    """Los 5 tubos: se apagan del más lejano al de la cabina, uno cada 0,30 s, con un parpadeo antes."""
    out = []
    for k in range(5):
        ta = T_APAGA + .30 * k                                   # fallan de a uno, con un parpadeo largo: se ve venir el apagón
        if t < ta - .30: v = 1.0
        elif t < ta: v = .2 if int(t * 22 + k) % 2 else 1.0
        else: v = 0.0
        out.append(v)
    return out


class Toma:
    INI, FIN = 52.05, 61.3          # de 61,3 a 62,3 la cara sigue debajo de la cortina diagonal (lo dibuja la T9)

    def preparar(s, E):
        LM.preparar(E); s.E = E
        s.W, s.H = int(os.environ.get('W_TP', 0)) or E.r.w, int(os.environ.get('H_TP', 0)) or E.r.h
        s.pos_press, s.yaw_press = CB.pose_presionar(LM.CABINA, LM.CAB_YAW)
        s.pista = caras([(-9, 54.0, 'decidido'), (54.0, 59.0, 'neutral'), (59.0, 99, 'desaprobacion')])
        if 'escaner_anillo' not in E.r.meshes:
            import t03; t03._anillo(E.r, 'escaner_anillo')

    def _max(s, t):
        E = s.E
        if t < T_ESCANEO:
            if not hasattr(s, 'M_press'):
                s.M_press = max_objs(E, 'tp_presionar_boton_1', 1.0, s.pos_press, s.yaw_press, suelo=PISO[2])[1]
                sp, n = CB.pantalla_pos(LM.CABINA, LM.CAB_YAW); s.p_toque = sp + n * .006
            w = suave((t - 52.9 + .30) / .30) * (1 - suave((t - 52.9 - .75) / .35))
            return max_objs(E, 'tp_presionar_boton_1', .85 + (t - 52.9), s.pos_press, s.yaw_press, pista=s.pista, t_escena=t, suelo=PISO[2],
                            ik=IK.varios(s._ik(t), IK.tocar(s.M_press, s.p_toque, w)))
        if t < T_CARA:
            return max_objs(E, 'g16_espera_respirando', 1 + (t - T_ESCANEO) * .7, PISO, LM.CAB_YAW, pista=s.pista, t_escena=t, suelo=PISO[2], ik=s._ik(t))
        if t > 60.85: E.max.cara_manual = {'eyeBlinkLeft': .35, 'eyeBlinkRight': .35}          # párpados caídos, resignado
        return max_objs(E, 'tp_parado_celular_1', 1.2 + (t - T_CARA) * .5, PISO, LM.CAB_YAW, pista=s.pista, t_escena=t, suelo=PISO[2], ik=s._ik(t))

    def _ik(s, t):
        """La mano izquierda rodea el celular; en el primer plano final la pantalla mira a su cara."""
        def fn(m, G, Pm, b):
            if t >= T_CARA:
                U, Fw, Rt = IK.ejes(m, G, b, 'Chest'); pc = Pm[m.idx['Chest']]; Uh, Fh, Rh = IK.ejes_cabeza(m, G, b)
                Pb, nb = IK.punto_cara(m, G, Pm, b, 'boca')
                W = pc + Fw * .30 - Rt * .10 - U * .13
                F = IK.marco_xz(U * .85 + Fw * .35 + Rt * .1, (Pb - W))               # el largo del celular hacia arriba, la palma a la cara
                IK.brazo(m, G, Pm, b, 'Left', W, F, polo=-U * .8 - Rt * .5, w=1.0)
            IK.dedos(m, G, Pm, b, 'Left', {'Index': (22, 30, 20), 'Middle': (26, 34, 22), 'Ring': (28, 36, 24), 'Pinky': (32, 38, 26)}, pulgar=(22, 6, 12, 14))
        return fn

    def _celular(s, M):
        """Celular por código (sin marcas): cuerpo negro con la pantalla encendida, acostado en la palma izquierda."""
        W = mano_mat(s.E, M, 'Left'); pl = IK.palma_local('Left', .095, .013)
        p = W[:3, 3] + W[:3, :3] @ pl; R = W[:3, :3]
        Mc = np.eye(4, dtype=np.float32); Mc[:3, :3] = R @ np.diag([.150, .073, .009]); Mc[:3, 3] = p
        Ms = np.eye(4, dtype=np.float32); Ms[:3, :3] = R @ np.diag([.138, .064, .002]); Ms[:3, 3] = p + R @ np.array([0, 0, .0052])
        return [dict(mesh='box', model=Mc, texture='white', color=(.04, .04, .05), roughness=.25),
                dict(mesh='box', model=Ms, texture='white', color=(.55, .65, .85), emission=1.4, noshadow=True, roughness=.1)], p, W

    def _pantalla(s, t):
        if t < T_ESCANEO:
            toque = max(0.0, min(1.0, (t - 52.9) / .5)) if t > 52.9 else 0.0
            im = PT.casita(1.0, None, toque)
        else:
            im = PT.casita(1.0, min(1.0, (t - T_ESCANEO - .2) / 1.9))
        PT.poner(s.E.r, 'cab_pant_t8', im)

    def cuadro(s, E, t):
        o, M, J = s._max(t)
        g = 1 - suave((t - T_ESCANEO - .2) / 1.7)                             # llega verde de México; el escáner lo va limpiando
        if g > 0:
            piel = {f'max_m{i}' for i, q in enumerate(E.max.prims) if q['mname'].startswith('Std_Skin')}
            for d in o:
                if d['mesh'] in piel: d['color'] = (1 - .22 * g, 1.0, 1 - .30 * g)
        cel, pc, W = s._celular(M)
        luces = luces_en(t); on = sum(luces) / 5
        s._pantalla(t)
        objs = LM.escena(E, t, luces, ventana=1.0, pantalla='cab_pant_t8', luz_cabina=1.0 if t < T_APAGA + .7 else 0.0) + o + cel
        extra = []
        if T_ESCANEO + .2 < t < T_GENERAL - .1:                             # el anillo verde sube de los pies a la cabeza
            z = 2.0 * min(1.0, (t - T_ESCANEO - .2) / 1.9)
            Ma = np.eye(4, dtype=np.float32); Ma[:3, 3] = PISO + (0, 0, z)
            objs.append(dict(mesh='escaner_anillo', model=Ma, texture='white', color=(.4, 1, .55), emission=8, noshadow=True))
            extra.append(dict(pos=tuple(PISO + (0, 0, z)), col=(.15, .6, .25)))
        if t >= T_CARA:                                                      # la luz del celular le alumbra la cara desde abajo
            extra.append(dict(pos=tuple(pc + (0, 0, .05) + F * .05), col=(.55, .58, .65)))
        if t < T_ESCANEO:
            sp, n = CB.pantalla_pos(LM.CABINA, LM.CAB_YAW); u = (t - s.INI) / (T_ESCANEO - s.INI)
            lat = -np.cross(n, [0, 0, 1.0])
            cam = (tuple(sp + n * .44 - lat * .40 + (0, 0, .12 - .03 * u)), tuple(sp + (0, 0, -.02)), 46 - 8 * u)
        elif t < T_GENERAL:
            u = (t - T_ESCANEO) / (T_GENERAL - T_ESCANEO)
            cam = (tuple(PISO + F * (2.75 - .3 * u) + S * .35 + (0, 0, 1.15)), tuple(PISO + (0, 0, 1.06)), 44)       # entra entero, con la cabeza
        elif t < T_NEGRO:
            u = (t - T_GENERAL) / (T_NEGRO - T_GENERAL)
            v = suave(u)                                                    # se acerca despacio mientras los tubos van fallando
            cam = (tuple(PISO + F * (4.3 - 1.9 * v) + S * (1.3 - .8 * v) + (0, 0, 1.20 + .15 * v)), tuple(PISO + (0, 0, 1.45 - .15 * v)), 56 - 10 * v)
        else:
            h = J['Head']; u = (t - T_CARA) / (s.FIN - T_CARA)
            cam = (tuple(h + F * (.85 - .06 * u) + (0, 0, -.06)), tuple(h + (0, 0, -.10)), 40)
        if T_NEGRO <= t < T_CARA:
            return np.zeros((s.H, s.W, 3), np.uint8)
        E.posts = []
        Lz = LM.luz(cam[1], cam[0], luces, 1.0 if t < T_APAGA else .25, extra)
        if t >= T_CARA:
            Lz.update(env_k=.02, sun_col=(0, 0, 0))
        return E.render(objs, cam, Lz)
