"""Personajes secundarios realistas (Rocketbox Bip01 / Mixamo) movidos con los BVH de Kimodo (esqueleto SOMA).

Retarget por ORIENTACIÓN con corrección de reposo (sirve aunque uno esté en T y el otro en A):
  para cada hueso b: C_b = giro mínimo que lleva la dirección de reposo del destino a la del BVH;
  giro global nuevo = G_bvh(t) · C_b · G_reposo_destino.
Así la dirección de cada hueso es exactamente la del BVH y además se copia el giro sobre su eje (muñeca, cabeza).
Los largos son los del personaje; la cadera se mueve escalada por la altura de cadera.

Uso:
    A = ActorRB(E.r, 'rb_mujer_f08', 'esposa', alto=1.64)
    objs = A.objetos(clip, t, pos=(x, y, z), giro=grados)      # giro 0 = mira -Y (igual que Max con yaw 0)
    A.hueso('Bip01 R Hand')                                    # posición mundo tras el último objetos()
"""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/fbxtool')
import gltf, skin

OBJ = '/home/claude/recursos_repo/recursos/out/objaverse'


def _mapa_bip():
    m = {'Hips': 'Bip01 Pelvis', 'Spine1': 'Bip01 Spine', 'Spine2': 'Bip01 Spine1', 'Chest': 'Bip01 Spine2',
         'Neck1': 'Bip01 Neck', 'Head': 'Bip01 Head'}
    for S, b in (('Left', 'L'), ('Right', 'R')):
        m.update({f'{S}Shoulder': f'Bip01 {b} Clavicle', f'{S}Arm': f'Bip01 {b} UpperArm', f'{S}ForeArm': f'Bip01 {b} Forearm',
                  f'{S}Hand': f'Bip01 {b} Hand', f'{S}Leg': f'Bip01 {b} Thigh', f'{S}Shin': f'Bip01 {b} Calf',
                  f'{S}Foot': f'Bip01 {b} Foot', f'{S}ToeBase': f'Bip01 {b} Toe0'})
        for k, f in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
            base = 1 if f == 'Thumb' else 2                       # SOMA Index1 = metacarpo
            for j in range(3):
                m[f'{S}Hand{f}{base + j}'] = f'Bip01 {b} Finger{k}' + ('' if j == 0 else str(j))
    return m


def _mapa_mixamo():
    t = {'Hips': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1', 'Chest': 'Spine2', 'Neck1': 'Neck', 'Head': 'Head'}
    for S in ('Left', 'Right'):
        for a, b in (('Shoulder', 'Shoulder'), ('Arm', 'Arm'), ('ForeArm', 'ForeArm'), ('Hand', 'Hand'),
                     ('Leg', 'UpLeg'), ('Shin', 'Leg'), ('Foot', 'Foot'), ('ToeBase', 'ToeBase')):
            t[S + a] = S + b
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            base = 1 if f == 'Thumb' else 2
            for j in range(3): t[f'{S}Hand{f}{base + j}'] = f'{S}Hand{f}{j + 1}'
    return {k: 'mixamorig:' + v for k, v in t.items()}


# hijo SOMA que da la dirección de cada hueso
def _hijo(b):
    cad = {'Hips': None, 'Spine1': 'Spine2', 'Spine2': 'Chest', 'Chest': 'Neck1', 'Neck1': 'Head', 'Head': 'HeadEnd'}
    if b in cad: return cad[b]
    for S in ('Left', 'Right'):
        ch = {f'{S}Shoulder': f'{S}Arm', f'{S}Arm': f'{S}ForeArm', f'{S}ForeArm': f'{S}Hand', f'{S}Hand': f'{S}HandMiddle2',
              f'{S}Leg': f'{S}Shin', f'{S}Shin': f'{S}Foot', f'{S}Foot': f'{S}ToeBase', f'{S}ToeBase': f'{S}ToeEnd'}
        if b in ch: return ch[b]
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            n0 = 1 if f == 'Thumb' else 2
            for j in range(3):
                if b == f'{S}Hand{f}{n0 + j}':
                    return f'{S}Hand{f}{n0 + j + 1}' if j < 2 else f'{S}Hand{f}End'
    return None


def _giro(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = float(np.dot(a, b)); s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0: return np.eye(3)
        p = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0]); p /= np.linalg.norm(p); return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


class RetargetSOMA:
    def __init__(s, dst, rest_src):
        """dst: gltf.Skinned; rest_src: un Clip cualquiera (para el reposo SOMA: nombres y posiciones)."""
        s.d = dst; J = dst.J; nodos = J['nodes']
        s.n2i = {n.get('name', ''): i for i, n in enumerate(nodos)}
        bip = 'Bip01 Pelvis' in s.n2i
        s.MAP = {k: v for k, v in (_mapa_bip() if bip else _mapa_mixamo()).items() if v in s.n2i}
        s.hijos = {}
        for c, p in dst.parent.items(): s.hijos.setdefault(p, []).append(c)
        s.Lr = {}
        for i, (T, R, S, M) in enumerate(dst.rest):
            if M is not None: L = np.array(M, float).reshape(4, 4).T
            else: L = np.eye(4); L[:3, :3] = gltf.q2m(R) @ np.diag(S); L[:3, 3] = T
            s.Lr[i] = L
        s.Gr = {}
        for i in range(len(nodos)): s._gr(i)
        s.raices = [i for i in range(len(nodos)) if i not in dst.parent]
        rp = rest_src.reposo; ix = rest_src.idx
        s.C = {}
        for sb, db in s.MAP.items():
            ch = _hijo(sb); i = s.n2i[db]
            if ch is None or ch not in ix: continue
            ds = rp[ix[ch]] - rp[ix[sb]]
            # dirección destino: al hueso mapeado del hijo, si existe; si no, al primer hijo del nodo
            dc = s.MAP.get(ch)
            if dc is not None: dt = s.Gr[s.n2i[dc]][:3, 3] - s.Gr[i][:3, 3]
            else:
                hs = s.hijos.get(i, [])
                if not hs: continue
                dt = s.Gr[hs[0]][:3, 3] - s.Gr[i][:3, 3]
            if np.linalg.norm(ds) < 1e-6 or np.linalg.norm(dt) < 1e-6: continue
            s.C[i] = _giro(dt, ds)
        # cadera: se usan los ejes (izq-der de las piernas, arriba de la columna) para su giro de reposo
        s.C[s.n2i[s.MAP['Hips']]] = s._corr_marco(rp, ix, 'Hips')
        s.C[s.n2i[s.MAP['Head']]] = s.C[s.n2i[s.MAP['Neck1']]]       # cabeza: misma corrección que el cuello (su primer hijo es un hueso de la cara)
        s.hip_src = rp[ix['Hips']][1]; s.hip_dst = s.Gr[s.n2i[s.MAP['Hips']]][1, 3]
        s.pelvis = s.n2i[s.MAP['Hips']]
        s.inv = {v: k for k, v in s.MAP.items()}
        # eje local con el que cada falange se CIERRA hacia la palma (en el reposo en T las palmas miran abajo: cerrar = bajar la punta)
        s.eje_dedo = {}
        for lado in ('L', 'R'):
            for k in range(5):
                prev = None
                for suf in ('', '1', '2'):
                    i = s.n2i.get(f'Bip01 {lado} Finger{k}{suf}')
                    if i is None: continue
                    hs = s.hijos.get(i, []); mejor = prev
                    if hs:
                        o = s.Lr[hs[0]][:3, 3]; mejor = None
                        for e in range(3):
                            for sg in (1, -1):
                                ax = np.zeros(3); ax[e] = sg
                                from scipy.spatial.transform import Rotation as Rot
                                y = (s.Gr[i][:3, :3] @ Rot.from_rotvec(ax * .5).as_matrix() @ o)[1]
                                if mejor is None or y < mejor[0]: mejor = (y, ax)
                    if mejor is not None: s.eje_dedo[i] = mejor[1]; prev = mejor

    def _corr_marco(s, rp, ix, b):
        def marco(izq, arr):
            e1 = izq / np.linalg.norm(izq); e2 = arr - e1 * np.dot(arr, e1); e2 /= np.linalg.norm(e2); return np.stack([e1, e2, np.cross(e1, e2)], 1)
        Fs = marco(rp[ix['LeftLeg']] - rp[ix['RightLeg']], rp[ix['Chest']] - rp[ix['Hips']])
        g = lambda k: s.Gr[s.n2i[s.MAP[k]]][:3, 3]
        Fd = marco(g('LeftLeg') - g('RightLeg'), g('Chest') - g('Hips'))
        return Fs @ Fd.T

    def _gr(s, i):
        if i in s.Gr: return s.Gr[i]
        p = s.d.parent.get(i); s.Gr[i] = (s._gr(p) @ s.Lr[i]) if p is not None else s.Lr[i]; return s.Gr[i]

    def locales(s, clip, t, hip0=None, capa=None, mirar=None):
        Gs, Ps = clip.globales(t); ix = clip.idx; LO = {}
        if capa is not None and capa[0] == 'saludo':     # saludo hecho por código: codo abajo y al lado, antebrazo arriba, la mano se mece
            _, fase, lado, peso = capa; Gs = Gs.copy(); rp = clip.reposo
            Rc = Gs[ix['Hips']]; U = np.array([0, 1.0, 0]); Fw = Rc @ np.array([0, 0, 1.0]); Fw[1] = 0; Fw /= np.linalg.norm(Fw)
            Rt = np.cross(Fw, U) * (1 if lado == 'Right' else -1)              # hacia afuera por ese lado
            Rt = -np.cross(U, Fw) if lado == 'Right' else np.cross(U, Fw)
            mece = math.sin(fase)
            obj = {lado + 'Arm': (lado + 'ForeArm', Rt * .62 - U * .62 + Fw * .48),
                   lado + 'ForeArm': (lado + 'Hand', U * .88 + Rt * (.18 + .30 * mece) + Fw * .30),
                   lado + 'Hand': (lado + 'HandMiddle2', U * .92 + Rt * (.10 + .55 * mece) + Fw * .15)}
            from scipy.spatial.transform import Rotation as Rot, Slerp
            for n, (hijo, d) in obj.items():
                i = ix[n]; Rn = _giro(rp[ix[hijo]] - rp[i], d)
                pw = peso if n == lado + 'Arm' else min(1.0, peso * 1.8)        # el antebrazo sube antes: la mano no sale abierta hacia el costado
                Rn = Slerp([0, 1], Rot.from_matrix([Gs[i], Rn]))(pw).as_matrix() if pw < 1 else Rn
                dif = Rn @ Gs[i].T
                if n == lado + 'Hand':
                    for nn, k in ix.items():
                        if nn.startswith(lado + 'Hand') and nn != n: Gs[k] = dif @ Gs[k]
                Gs[i] = Rn
        elif capa is not None:                   # brazo de otro clip, relativo al pecho (p. ej. saludar con la taza en la otra mano)
            c2, t2, lado, peso = capa
            G2, _ = c2.globales(t2); Gs = Gs.copy(); ch = ix['Chest']
            Rr = Gs[ch] @ G2[ch].T
            for n, i in ix.items():
                if n.startswith(lado) and any(k in n for k in ('Shoulder', 'Arm', 'Hand')):
                    Rn = Rr @ G2[i]
                    if peso < 1:
                        from scipy.spatial.transform import Rotation as Rot, Slerp
                        Rn = Slerp([0, 1], Rot.from_matrix([Gs[i], Rn]))(peso).as_matrix()
                    Gs[i] = Rn
        if mirar is not None:                       # gira cuello y cabeza hacia una dirección (marco del BVH): el cuello 40 %, la cabeza el resto
            from scipy.spatial.transform import Rotation as Rot
            d, peso = mirar; Gs = Gs.copy(); d = np.asarray(d, float) / np.linalg.norm(d)
            f = Gs[ix['Head']] @ np.array([0, 0, 1.0]); f /= np.linalg.norm(f)
            rv = Rot.from_matrix(_giro(f, d)).as_rotvec() * peso
            ang = np.linalg.norm(rv)
            if ang > math.radians(55): rv = rv / ang * math.radians(55)                 # nunca más de 55 grados
            for n_, k_ in (('Neck1', .4), ('Head', 1.0)):
                if n_ in ix: Gs[ix[n_]] = Rot.from_rotvec(rv * k_).as_matrix() @ Gs[ix[n_]]
        def visita(i, Gp):
            Gt = Gp @ s.Lr[i]; sb = s.inv.get(s.d.J['nodes'][i].get('name', ''))
            if sb is not None and i in s.C:
                G = Gt.copy(); G[:3, :3] = Gs[ix[sb]] @ s.C[i] @ s.Gr[i][:3, :3]
                if i == s.pelvis:
                    d = Ps[ix['Hips']] - (hip0 if hip0 is not None else clip.reposo[ix['Hips']])
                    G[:3, 3] = s.Gr[i][:3, 3] + d * (s.hip_dst / s.hip_src)
                Gt = G; LO[i] = np.linalg.inv(Gp) @ Gt
            for c in s.hijos.get(i, []): visita(c, Gt)
        for r in s.raices: visita(r, np.eye(4))
        return LO


# caras (formas ARKit 'AK_' de Rocketbox), pesos probados en el casino
EXPR = {
    'neutral': {},
    'alegria': {'AK_44_MouthSmileLeft': 1.0, 'AK_45_MouthSmileRight': 1.0, 'AK_25_JawOpen': .25, 'AK_07_CheekSquintLeft': .6, 'AK_08_CheekSquintRight': .6, 'AK_03_BrowInnerUp': .3},
    'sorpresa': {'AK_21_EyeWideLeft': 1.0, 'AK_22_EyeWideRight': 1.0, 'AK_03_BrowInnerUp': 1.0, 'AK_04_BrowOuterUpLeft': .9, 'AK_05_BrowOuterUpRight': .9, 'AK_25_JawOpen': .45},
    'triste': {'AK_30_MouthFrownLeft': 1.0, 'AK_31_MouthFrownRight': 1.0, 'AK_03_BrowInnerUp': 1.1, 'AK_11_EyeLookDownLeft': .4, 'AK_12_EyeLookDownRight': .4, 'AU_41_LidDroop': .4},
    'molesto': {'AK_01_BrowDownLeft': 1.1, 'AK_02_BrowDownRight': 1.1, 'AK_30_MouthFrownLeft': .6, 'AK_31_MouthFrownRight': .6, 'AK_36_MouthPressLeft': .6, 'AK_37_MouthPressRight': .6, 'AK_50_NoseSneerLeft': .3, 'AK_51_NoseSneerRight': .3},
    'susto': {'AK_21_EyeWideLeft': 1.2, 'AK_22_EyeWideRight': 1.2, 'AK_03_BrowInnerUp': 1.2, 'AK_25_JawOpen': .7, 'AK_46_MouthStretchLeft': .6, 'AK_47_MouthStretchRight': .6},
    'sueno': {'AU_41_LidDroop': .9, 'AK_25_JawOpen': .9, 'AK_32_MouthFunnel': .3, 'AK_03_BrowInnerUp': .5, 'AK_19_EyeSquintLeft': .5, 'AK_20_EyeSquintRight': .5},
}


def mezcla(*pares):
    out = {}
    for w, e in pares:
        if w <= 0: continue
        for k, v in (EXPR[e] if isinstance(e, str) else e).items(): out[k] = out.get(k, 0) + w * v
    return out


def parpadeo(f, t, cada=3.1, fase=.4, dur=.09):
    tb = round((t - fase) / cada) * cada + fase; k = max(0.0, 1 - abs(t - tb) / dur)
    if k:
        f = dict(f); f['AK_09_EyeBlinkLeft'] = f.get('AK_09_EyeBlinkLeft', 0) + k; f['AK_10_EyeBlinkRight'] = f.get('AK_10_EyeBlinkRight', 0) + k
    return f


class ActorRB:
    def __init__(s, r, nombre, clave, alto=1.65, clip_reposo=None, rugosidad=.6):
        from soma import Clip
        s.r = r; s.clave = clave; s.S = gltf.Skinned(f'{OBJ}/{nombre}.glb', alto); s.rug = rugosidad
        s.rt = RetargetSOMA(s.S, clip_reposo or Clip('/home/claude/teletransporte/anims/tp_abanicarse_1.bvh'))
        s.tex, s.uv = [], []
        for i, p in enumerate(s.S.prims):
            s.uv.append(np.stack([p['UV'][p['idx']][:, 0], 1 - p['UV'][p['idx']][:, 1]], 1).astype(np.float32))
            if p['image'] is not None:
                k = f'{clave}_t{i}'; im = p['image'].convert('RGBA')
                if max(im.size) > 2048: im = im.resize((2048, 2048))
                r.add_texture(k, im); r.set_texture_filter(k, 'linear'); s.tex.append(k)
            else: s.tex.append(None)
        s.M = np.eye(4)

    def malla(s, clip, t, hip0='inicio', capa=None, dedos=None, mirar=None):
        Ps0 = clip.globales(0)[1][clip.idx['Hips']].copy()
        if hip0 == 'inicio': h0 = np.array([Ps0[0], clip.reposo[clip.idx['Hips']][1], Ps0[2]])
        else: h0 = None
        LO = s.rt.locales(clip, t, h0, capa, mirar)
        if dedos:                                         # {'L': grados, 'R': grados}: cierra los dedos (el pulgar, la mitad)
            from scipy.spatial.transform import Rotation as Rot
            for lado, ang in dedos.items():
                for k in range(5):
                    for suf in ('', '1', '2'):
                        i = s.rt.n2i.get(f'Bip01 {lado} Finger{k}{suf}')
                        if i is None or i not in s.rt.eje_dedo: continue
                        an = ang.get(k, 0.0) if isinstance(ang, dict) else ang                # un número para todos, o {dedo: grados}
                        a = math.radians(an * (.45 if k == 0 else 1.0) * (1.0 if suf != '2' else .7))
                        Lm = LO.get(i, s.rt.Lr[i]).copy(); Lm[:3, :3] = Lm[:3, :3] @ Rot.from_rotvec(s.rt.eje_dedo[i] * a).as_matrix(); LO[i] = Lm
        return s.S.pose(None, 0, local_override=LO)

    def objetos(s, clip, t, pos, giro=0.0, pies_al_piso=False, cara=None, parpadea=True, capa=None, dedos=None, M_extra=None, mirar=None):
        f = EXPR[cara] if isinstance(cara, str) else (cara or {})
        if parpadea: f = parpadeo(f, t)
        if f != getattr(s, '_cara', None): s.S.set_morph(f); s._cara = f
        res = s.malla(clip, t, capa=capa, dedos=dedos, mirar=mirar)
        a = math.radians(giro)                            # glTF +Z (frente) -> -Y del mundo: giro 0 = mira -Y como Max
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.]])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pos
        if M_extra is not None: M = np.asarray(M_extra, float) @ M
        if pies_al_piso:
            V = np.concatenate([P for P, _ in res]); M[2, 3] -= float(V[:, 2].min())
        s.M = M; o = []
        V = np.concatenate([P for P, _ in res]); s.ult_V = V @ M[:3, :3].T + M[:3, 3]; s.ult_zmin = float(s.ult_V[:, 2].min())
        import real_scene as RS
        for i, ((P, N), p) in enumerate(zip(res, s.S.prims)):
            if not len(P): continue
            nm = f'{s.clave}_m{i}'
            RS.set_mesh(s.r, nm, skin.interleave(P.astype(np.float32), N.astype(np.float32), s.uv[i]))
            d = dict(mesh=nm, model=M.astype(np.float32), texture=s.tex[i] or 'white', roughness=s.rug)
            d['color'] = (1, 1, 1) if s.tex[i] else tuple(float(c) for c in p['color'][:3])
            if p.get('alpha', 'OPAQUE') != 'OPAQUE' or 'opacity' in p.get('mname', '').lower(): d['transparent'] = True
            o.append(d)
        return o

    def hueso_mat(s, nombre):
        """Matriz mundo (rotación sin escala + posición) de un hueso tras el último objetos(): para pegar objetos a la mano."""
        i = s.rt.n2i[nombre]; G = s.S.G[i]; Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        R = Cz @ G[:3, :3]; R = R / np.linalg.norm(R, axis=0, keepdims=True)
        p = ((Cz @ G[:3, 3]) - [0, 0, s.S.zoff]) * s.S.scale
        W = np.eye(4); W[:3, :3] = R; W[:3, 3] = p
        return s.M @ W

    def hueso(s, nombre):
        p = s.S.joint_pos(nombre); return (s.M @ np.r_[p, 1])[:3]
