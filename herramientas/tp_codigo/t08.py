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
        elif t < ta: v = .5 if int(t * 11 + k) % 2 else 1.0              # parpadeo suave (a 22 por segundo era un flash molesto)
        else: v = 0.0
        out.append(v)
    return out


CELULAR = (.062, .020, -.010, 0.0, (-25, -15, 0, 0))          # adelante, sobre la palma, hacia el meñique, cierre de dedos, pulgar


class Toma:
    INI, FIN = 52.05, 61.3          # de 61,3 a 62,3 la cara sigue debajo de la cortina diagonal (lo dibuja la T9)

    def preparar(s, E):
        LM.preparar(E); s.E = E
        s.W, s.H = int(os.environ.get('W_TP', 0)) or E.r.w, int(os.environ.get('H_TP', 0)) or E.r.h
        s.pos_press, s.yaw_press = CB.pose_presionar(LM.CABINA, LM.CAB_YAW)
        s.pista = caras([(-9, 54.0, 'decidido'), (54.0, 59.0, 'neutral'), (59.0, 99, 'desaprobacion')])
        if 'escaner_anillo' not in E.r.meshes:
            import t03; t03._anillo(E.r, 'escaner_anillo')
        # agarre del celular MEDIDO con la malla (herramientas/medir_celular.py): ningún dedo adentro, la piel tocándolo
        s.cel_adel, s.cel_hondo, s.cel_lado, s.cel_curva, s.cel_pulgar = CELULAR
        s.cel_R = np.eye(3); s._apoyar_celular()

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

    def _ik(s, t, curva=None):
        """La mano izquierda rodea el celular; en el primer plano final la pantalla mira a su cara."""
        def fn(m, G, Pm, b):
            if t >= T_CARA:
                U, Fw, Rt = IK.ejes(m, G, b, 'Chest'); pc = Pm[m.idx['Chest']]; Uh, Fh, Rh = IK.ejes_cabeza(m, G, b)
                Pb, nb = IK.punto_cara(m, G, Pm, b, 'boca')
                W = pc + Fw * .30 - Rt * .10 - U * .13
                F = IK.marco_xz(U * .85 + Fw * .35 + Rt * .1, (Pb - W))               # el largo del celular hacia arriba, la palma a la cara
                IK.brazo(m, G, Pm, b, 'Left', W, F, polo=-U * .8 - Rt * .5, w=1.0)
            a = s.cel_curva if curva is None else curva                          # cerrados lo justo: rodean el borde del celular sin atravesarlo
            IK.dedos(m, G, Pm, b, 'Left', {d: (a, a * 1.15, a * .8) for d in ('Index', 'Middle', 'Ring', 'Pinky')}, pulgar=s.cel_pulgar)
        return fn

    def _apoyar_celular(s):
        """El celular se apoya SOBRE la palma: se mide la superficie de la mano (malla real, dedos estirados) debajo de
        donde va, se inclina como ella y queda 1 mm por encima del punto más alto. Después se cierran los dedos lo justo."""
        E = s.E; t = T_CARA + .1
        o, M, J = max_objs(E, 'tp_parado_celular_1', 1.2 + (t - T_CARA) * .5, PISO, LM.CAB_YAW, suelo=PISO[2], ik=s._ik(t, 0))
        W = mano_mat(E, M, 'Left')
        Pw = np.concatenate([q for k, (q, _) in enumerate(E.max._ult_pose[2]) if E.max.prims[k]['mname'].startswith('Std_Skin_Arm')]) @ M[:3, :3].T + M[:3, 3]
        H = (Pw - W[:3, 3]) @ W[:3, :3]; H = H[np.linalg.norm(H, axis=1) < .13]          # solo la piel de la mano
        a0, l0 = s.cel_adel, s.cel_lado; hx, hy = .073 / 2, .150 / 2
        zona = (np.abs(H[:, 0] - a0) < hx + .004) & (np.abs(H[:, 1] - l0) < hy + .004)
        Hz = H[zona]; celdas = {}
        for v in Hz:
            k = (int((v[0] - a0) // .01), int((v[1] - l0) // .01)); celdas[k] = max(celdas.get(k, -9), v[2])
        A_ = np.array([[1, (k[0] + .5) * .01, (k[1] + .5) * .01] for k in celdas]); z_ = np.array(list(celdas.values()))
        c0, cx, cy = np.linalg.lstsq(A_, z_, rcond=None)[0]                         # plano de la palma: z = c0 + cx*x + cy*y
        n = np.array([-cx, -cy, 1.0]); n /= np.linalg.norm(n)
        x = np.array([1.0, 0, 0]) - n * n[0]; x /= np.linalg.norm(x); y = np.cross(n, x)
        Rl = np.stack([y, -x, n], 1)                                                 # caja: largo (.150) a lo ancho de la mano, ancho (.073) hacia los dedos
        base = np.array([a0, l0, c0])
        alto = float(((Hz - base) @ n).max())                                        # lo más alto de la mano bajo el celular
        s.cel_R = Rl; s.cel_c = base + n * (alto + .0045 + .001)
        for a in range(0, 91, 5):                                                    # dedos: lo más cerrados sin meterse
            if max(s._penetra(tt, a)[0] for tt in (T_CARA + .1, s.FIN - .15)) > .0008: break
            s.cel_curva = a
        s.cel_pen, s.cel_cont = s._penetra(T_CARA + .1, s.cel_curva)

    def _penetra(s, t, curva, adel=None, hondo=None):
        """Cuánto se meten los dedos (malla real de Max) dentro del celular, y a qué distancia queda la piel más cercana."""
        E = s.E
        o, M, J = max_objs(E, 'tp_parado_celular_1', 1.2 + (t - T_CARA) * .5, PISO, LM.CAB_YAW, suelo=PISO[2], ik=s._ik(t, curva))
        cel, p, W = s._celular(M); Mc = cel[0]['model'].astype(float)
        Pl = np.concatenate([q for q, _ in E.max._ult_pose[2]]); Pw = Pl @ M[:3, :3].T + M[:3, 3]
        Pw = Pw[np.linalg.norm(Pw - p, axis=1) < .14]
        L = (Pw - Mc[:3, 3]) @ np.linalg.inv(Mc[:3, :3]).T                                # en unidades de la caja (-.5..5)
        h = np.array([.150, .073, .009]) / 2; Ld = np.abs(L) * np.array([.150, .073, .009])
        dentro = np.all(Ld < h, axis=1); fuera = np.linalg.norm(np.maximum(Ld - h, 0), axis=1)
        pen = float((h - Ld)[dentro].min(axis=1).max()) if dentro.any() else 0.0
        return pen, float(fuera.min())

    def _medir_celular(s):
        """Busca dónde va el celular en la palma y cuánto cerrar los dedos: ningún dedo adentro (< 1 mm) y la piel tocándolo."""
        mejor = None
        for adel in (.050, .060, .070):
            for hondo in (.011, .015):
                for a in range(0, 76, 5):
                    pen = max(s._penetra(t, a, adel, hondo)[0] for t in (T_CARA + .1, s.FIN - .2))
                    if pen < .001:
                        cont = s._penetra(T_CARA + .1, a, adel, hondo)[1]
                        nota = a - 400 * cont                                              # lo más cerrado posible, pegado al celular
                        if mejor is None or nota > mejor[0]: mejor = (nota, adel, hondo, a, pen, cont)
        _, s.cel_adel, s.cel_hondo, s.cel_curva, s.cel_pen, s.cel_cont = mejor

    def _celular(s, M):
        """Celular por código (sin marcas): cuerpo negro con la pantalla encendida, acostado en la palma izquierda."""
        W = mano_mat(s.E, M, 'Left'); p = W[:3, 3] + W[:3, :3] @ s.cel_c
        R = W[:3, :3] @ s.cel_R                                                     # apoyado en la palma (medido), el largo cruza la mano
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
