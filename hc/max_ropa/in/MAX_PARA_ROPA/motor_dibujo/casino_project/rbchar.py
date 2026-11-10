"""Personaje Rocketbox (cara animable) con animaciones Mixamo retargeteadas, expresiones y mirada a un punto."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/fbxtool')
import gltf, retarget, skin
import real_scene as RS
from effects import minimum_jerk as mj
REC = '/home/claude/casino2/rec/all'

EXPR = {
    'malvada': {'AK_44_MouthSmileLeft': 1.35, 'AK_45_MouthSmileRight': .8, 'AK_01_BrowDownLeft': 1.3, 'AK_02_BrowDownRight': 1.3, 'AK_19_EyeSquintLeft': 1.0, 'AK_20_EyeSquintRight': 1.0,
                'AK_07_CheekSquintLeft': .9, 'AK_50_NoseSneerLeft': .5, 'AU_12_L_LipCornerPuller': .8, 'AK_33_MouthLeft': .3},
    'carcajada': {'AK_44_MouthSmileLeft': 1.2, 'AK_45_MouthSmileRight': 1.2, 'AK_25_JawOpen': .75, 'AK_07_CheekSquintLeft': 1.0, 'AK_08_CheekSquintRight': 1.0,
                  'AK_19_EyeSquintLeft': .9, 'AK_20_EyeSquintRight': .9, 'AK_01_BrowDownLeft': .5, 'AK_02_BrowDownRight': .5, 'AK_48_MouthUpperUpLeft': .4, 'AK_49_MouthUpperUpRight': .4},
    'molesto': {'AK_01_BrowDownLeft': 1.4, 'AK_02_BrowDownRight': 1.4, 'AK_30_MouthFrownLeft': .8, 'AK_31_MouthFrownRight': .8, 'AK_36_MouthPressLeft': .7, 'AK_37_MouthPressRight': .7,
                'AK_19_EyeSquintLeft': .4, 'AK_20_EyeSquintRight': .4, 'AK_50_NoseSneerLeft': .4, 'AK_51_NoseSneerRight': .4},
    'triste': {'AK_30_MouthFrownLeft': 1.3, 'AK_31_MouthFrownRight': 1.3, 'AK_03_BrowInnerUp': 1.4, 'AK_11_EyeLookDownLeft': .5, 'AK_12_EyeLookDownRight': .5,
               'AK_42_MouthShrugLower': .6, 'AU_41_LidDroop': .5},
    'sorpresa': {'AK_21_EyeWideLeft': 1.2, 'AK_22_EyeWideRight': 1.2, 'AK_03_BrowInnerUp': 1.2, 'AK_04_BrowOuterUpLeft': 1.0, 'AK_05_BrowOuterUpRight': 1.0, 'AK_25_JawOpen': .6},
    'alegria': {'AK_44_MouthSmileLeft': 1.3, 'AK_45_MouthSmileRight': 1.3, 'AK_25_JawOpen': .4, 'AK_07_CheekSquintLeft': .8, 'AK_08_CheekSquintRight': .8, 'AK_03_BrowInnerUp': .5},
}
def mix(*pairs):
    out = {}
    for w, e in pairs:
        if w <= 0: continue
        for k, v in (EXPR[e] if isinstance(e, str) else e).items(): out[k] = out.get(k, 0) + w * v
    return out
def blink(f, t, times, dur=.09):
    for tb in times:
        k = max(0, 1 - abs(t - tb) / dur)
        if k:
            f['AK_09_EyeBlinkLeft'] = f.get('AK_09_EyeBlinkLeft', 0) + k; f['AK_10_EyeBlinkRight'] = f.get('AK_10_EyeBlinkRight', 0) + k
    return f

class RBChar:
    def __init__(s, r, key, glb, height, anims):
        s.r = r; s.key = key; s.d = gltf.Skinned(f'{REC}/expresiones/{glb}.glb', height); s.tex = []
        for i, p in enumerate(s.d.prims):
            t = 'white'
            if p['image'] is not None:
                t = f'{key}_t{i}'; im = p['image'].convert('RGBA')
                if max(im.size) > 1024: im = im.resize((1024, 1024))
                r.add_texture(t, im); r.set_texture_filter(t, 'linear')
            s.tex.append(t)
        s.A = {}
        for n, path in anims.items():
            s.A[n] = (retarget.Retarget(f'{REC}/{path}', s.d), skin.Anim(f'{REC}/{path}'))
    def g2z(s, p):   # glTF -> mundo Z-arriba (sin la matriz de colocación)
        p = np.asarray(p, float); return (np.array((p[0], -p[2], p[1])) - (0, 0, s.d.zoff)) * s.d.scale
    def z2g(s, p):
        z = np.asarray(p, float) / s.d.scale + np.array((0, 0, s.d.zoff)); return np.array((z[0], z[2], -z[1]))
    def world_mat(s, jname, M=np.eye(4)):
        """Matriz mundo (Z-arriba, metros) de un hueso tras el último pose: para pegar objetos rígidamente."""
        i = s.A[next(iter(s.A))][0].name2node[jname]; G = s.d.G[i]
        C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        W = np.eye(4); W[:3, :3] = C @ G[:3, :3]; W[:3, :3] /= np.linalg.norm(W[:3, :3], axis=0, keepdims=True)
        W[:3, 3] = (C @ G[:3, 3] - (0, 0, s.d.zoff)) * s.d.scale
        return np.asarray(M) @ W
    def curl(s, LO, rt, side, amounts, axis=2):
        """Cierra dedos: amounts = {dedo(0..4): grados}; gira cada falange sobre su eje local."""
        for k, ang in amounts.items():
            for j in ('', '1', '2'):
                n = f'Bip01 {side} Finger{k}{j}'
                if n not in rt.name2node: continue
                i = rt.name2node[n]; L = LO.get(i, rt.L_rest[i]).copy()
                a = np.radians(ang * (1.0 if j == '' else .9)); c, s_ = np.cos(a), np.sin(a)
                R = np.eye(3)
                if axis == 2: R[:2, :2] = [[c, -s_], [s_, c]]
                elif axis == 1: R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s_, -s_, c
                else: R[1:, 1:] = [[c, -s_], [s_, c]]
                L[:3, :3] = L[:3, :3] @ R; LO[i] = L
        return LO
    def pose(s, layers, morph, look=None, look_w=.7, M=np.eye(4), curls=None, curl_axis=2, post=None):
        """layers: [(anim, tiempo, peso)] mezclados por vértices; look: punto mundo Z-arriba al que mira."""
        s.d.set_morph(morph); res = None; Minv = np.linalg.inv(M); s.joints = {}
        tot = sum(w for _, _, w in layers if w > 0)
        for name, t, w in layers:
            if w <= 0: continue
            rt, A = s.A[name]; LO = rt.locals(A, t)
            for side, am in (curls or {}).items(): s.curl(LO, rt, side, am, curl_axis)
            if post: post(LO, rt)
            if look is not None:
                lp = (Minv @ np.r_[look, 1])[:3]; rt.look_at(LO, s.z2g(lp), look_w)
            rr = s.d.pose(None, 0, local_override=LO); w = w / tot
            for jn in ('Bip01 Head', 'Bip01 R Hand', 'Bip01 L Hand', 'Bip01 R Finger2', 'Bip01 L Finger2', 'Bip01 Pelvis', 'Bip01 R Finger1', 'Bip01 R Finger11', 'Bip01 R Finger21', 'Bip01 R Finger0', 'Bip01 R Finger01',
                       'Bip01 L Finger1', 'Bip01 L Finger11', 'Bip01 L Finger0', 'Bip01 L Finger01', 'Bip01 MUpperLip', 'Bip01 REye', 'Bip01 LEye'):
                s.joints[jn] = s.joints.get(jn, 0) + w * (M @ np.r_[s.d.joint_pos(jn), 1])[:3]
            res = [(w * P, w * N) for P, N in rr] if res is None else [(P0 + w * P, N0 + w * N) for (P0, N0), (P, N) in zip(res, rr)]
        s.res = res; return res
    def draw(s, objs, M=np.eye(4), white=False):
        for i, ((P, N), p) in enumerate(zip(s.res, s.d.prims)):
            UV = p['UV'][p['idx']]; UV = np.stack([UV[:, 0], 1 - UV[:, 1]], 1)
            RS.set_mesh(s.r, f'{s.key}_m{i}', skin.interleave(P, N, UV))
            objs.append(dict(mesh=f'{s.key}_m{i}', model=np.asarray(M, np.float32), texture='white' if white else s.tex[i], color=(1, 1, 1), roughness=.55, skin=.5, ps1=0,
                             emission=30 if white else 0, transparent=(p['alpha'] != 'OPAQUE') and not white))

# ---------------- IK de brazo (dos huesos) ----------------
import retarget as _rt
def fk(rt, LO, i, memo):
    if i in memo: return memo[i]
    L = LO.get(i, rt.L_rest[i]); p = rt.dst.parent.get(i)
    memo[i] = (fk(rt, LO, p, memo) @ L) if p is not None else L; return memo[i]

def ik_arm(ch, rt, LO, side, target_z, pole_z, M=np.eye(4), finger_dir_z=None, twist=0.0, flick=0.0):
    """Coloca la muñeca en target (mundo Z-arriba) con el codo hacia pole. finger_dir_z: hacia dónde apuntan los dedos.
    flick: giro extra de la muñeca (grados) para abanicar."""
    Minv = np.linalg.inv(M); g = lambda p: ch.z2g((Minv @ np.r_[p, 1])[:3])
    n = rt.name2node; ua, fa, hd = n[f'Bip01 {side} UpperArm'], n[f'Bip01 {side} Forearm'], n[f'Bip01 {side} Hand']
    memo = {}; Gua = fk(rt, LO, ua, memo); Gfa = fk(rt, LO, fa, memo); Ghd = fk(rt, LO, hd, memo)
    S, E, W = Gua[:3, 3], Gfa[:3, 3], Ghd[:3, 3]; a = np.linalg.norm(E - S); b = np.linalg.norm(W - E)
    T = g(target_z); P = g(pole_z); dv = T - S; d = min(np.linalg.norm(dv), a + b - 1e-4); dirv = dv / np.linalg.norm(dv)
    cosA = np.clip((a * a + d * d - b * b) / (2 * a * d), -1, 1); sinA = np.sqrt(1 - cosA * cosA)
    pv = P - S; bend = pv - dirv * pv.dot(dirv); bend /= np.linalg.norm(bend)
    E2 = S + dirv * a * cosA + bend * a * sinA; T2 = S + dirv * d
    Gp = fk(rt, LO, rt.dst.parent[ua], memo)
    Gua2 = Gua.copy(); Gua2[:3, :3] = _rt.rot_between(E - S, E2 - S) @ Gua[:3, :3]; LO[ua] = np.linalg.inv(Gp) @ Gua2
    Gfa_t = Gua2 @ LO.get(fa, rt.L_rest[fa]); wl = np.linalg.inv(Gfa) @ np.r_[W, 1]; Wn = (Gfa_t @ wl)[:3]
    Gfa2 = Gfa_t.copy(); Gfa2[:3, :3] = _rt.rot_between(Wn - Gfa_t[:3, 3], T2 - Gfa_t[:3, 3]) @ Gfa_t[:3, :3]; LO[fa] = np.linalg.inv(Gua) @ Gfa2 if False else np.linalg.inv(Gua2) @ Gfa2
    Ghd_t = Gfa2 @ LO.get(hd, rt.L_rest[hd])
    if finger_dir_z is not None:
        f2 = n[f'Bip01 {side} Finger2']; fr = rt.G_rest[f2][:3, 3] - rt.G_rest[hd][:3, 3]
        f_now = Ghd_t[:3, :3] @ rt.G_rest[hd][:3, :3].T @ fr
        want = g(np.asarray(finger_dir_z) + target_z) - g(target_z)
        R = _rt.rot_between(f_now, want)
        Ghd2 = Ghd_t.copy(); Ghd2[:3, :3] = R @ Ghd_t[:3, :3]
    else: Ghd2 = Ghd_t.copy()
    if flick:
        ax = Gfa2[:3, :3] @ np.array((0, 0, 1.)); ax /= np.linalg.norm(ax); th = np.radians(flick)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); Rf = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        Ghd2[:3, :3] = Rf @ Ghd2[:3, :3]
    LO[hd] = np.linalg.inv(Gfa2) @ Ghd2
    return LO
