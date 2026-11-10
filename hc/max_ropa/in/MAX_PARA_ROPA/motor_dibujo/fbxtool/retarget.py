"""Retarget de animaciones Mixamo (FBX) a esqueletos Biped Bip01 (Rocketbox, glTF con caras animables).
Huesos de cadena: se igualan las DIRECCIONES de cada hueso (sirve aunque una pose de reposo sea T y la otra A).
Pelvis y cabeza: se copia el giro relativo al reposo. Traslación de cadera escalada por altura de cadera."""
import numpy as np
import skin

SIDE = {'Left': 'L', 'Right': 'R'}
def build_map():
    m = {'Hips': 'Bip01 Pelvis', 'Spine': 'Bip01 Spine', 'Spine1': 'Bip01 Spine1', 'Spine2': 'Bip01 Spine2', 'Neck': 'Bip01 Neck', 'Head': 'Bip01 Head'}
    for s, b in SIDE.items():
        m.update({f'{s}Shoulder': f'Bip01 {b} Clavicle', f'{s}Arm': f'Bip01 {b} UpperArm', f'{s}ForeArm': f'Bip01 {b} Forearm', f'{s}Hand': f'Bip01 {b} Hand',
                  f'{s}UpLeg': f'Bip01 {b} Thigh', f'{s}Leg': f'Bip01 {b} Calf', f'{s}Foot': f'Bip01 {b} Foot', f'{s}ToeBase': f'Bip01 {b} Toe0'})
        for k, f in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
            for j in range(3):
                m[f'{s}Hand{f}{j + 1}'] = f'Bip01 {b} Finger{k}' + ('' if j == 0 else str(j))
    return m
MAP = build_map()
# hueso -> hijo que define su dirección (en nombres Mixamo)
def child_of(b):
    order = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head']
    if b in order[1:-1]: return order[order.index(b) + 1]
    for s in SIDE:
        ch = {f'{s}Shoulder': f'{s}Arm', f'{s}Arm': f'{s}ForeArm', f'{s}ForeArm': f'{s}Hand', f'{s}Hand': f'{s}HandMiddle1',
              f'{s}UpLeg': f'{s}Leg', f'{s}Leg': f'{s}Foot', f'{s}Foot': f'{s}ToeBase'}
        if b in ch: return ch[b]
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            for j in (1, 2):
                if b == f'{s}Hand{f}{j}': return f'{s}Hand{f}{j + 1}'
    return None
ORIENT = {'Hips', 'Head'}

def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.9999:
        ax = np.cross(a, [1, 0, 0]);
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax); return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

class Retarget:
    def __init__(s, src_skel_path, dst, flip_y=False):
        s.src = skin.Character(src_skel_path); s.dst = dst; J = dst.J
        s.name2node = {n.get('name', ''): i for i, n in enumerate(J['nodes'])}
        s.children = {}
        for c, p in dst.parent.items(): s.children.setdefault(p, []).append(c)
        s.L_rest = {}
        for i, (T, R, S, M) in enumerate(dst.rest):
            if M is not None: L = np.array(M, float).reshape(4, 4).T
            else:
                L = np.eye(4); L[:3, :3] = __import__("gltf").q2m(R) @ np.diag(S); L[:3, 3] = T
            s.L_rest[i] = L
        s.G_rest = {}
        for i in range(len(J['nodes'])): s._grest(i)
        s.roots = [i for i in range(len(J['nodes'])) if i not in dst.parent]
        Gb = s.src.globals(); s.src_bind = {s.src.nodes[k]['name']: v for k, v in Gb.items()}
        # dirección del hijo en el marco local del hueso destino (reposo)
        s.off = {}
        for mb, db in MAP.items():
            ch = child_of(mb)
            if db not in s.name2node or ch is None or MAP.get(ch) not in s.name2node: continue
            i, c = s.name2node[db], s.name2node[MAP[ch]]
            s.off[i] = (np.linalg.inv(s.G_rest[i]) @ s.G_rest[c])[:3, 3]
        hs = s.src_bind['Hips'][:3, 3]; s.src_hip_h = hs[1]
        s.dst_hip_h = s.G_rest[s.name2node['Bip01 Pelvis']][1, 3]
        s.flip = np.diag([-1, 1, -1]) if flip_y else np.eye(3)
    def _grest(s, i):
        if i in s.G_rest: return s.G_rest[i]
        p = s.dst.parent.get(i); s.G_rest[i] = (s._grest(p) @ s.L_rest[i]) if p is not None else s.L_rest[i]; return s.G_rest[i]
    def locals(s, anim, t):
        Gs = s.src.globals(anim, t); src = {s.src.nodes[k]['name']: v for k, v in Gs.items()}
        F = s.flip; LO = {}; Gn = {}; inv = {v: k for k, v in MAP.items()}
        def visit(i, Gp):
            name = s.dst.J['nodes'][i].get('name', ''); mb = inv.get(name); L = s.L_rest[i]
            Gt = Gp @ L
            if mb in src:
                if mb in ORIENT:
                    dR = F @ src[mb][:3, :3] @ s.src_bind[mb][:3, :3].T @ F.T
                    G = Gt.copy(); G[:3, :3] = dR @ s.G_rest[i][:3, :3]
                    if mb == 'Hips':
                        d = F @ (src['Hips'][:3, 3] - s.src_bind['Hips'][:3, 3]) * (s.dst_hip_h / s.src_hip_h)
                        G[:3, 3] = s.G_rest[i][:3, 3] + d
                    Gt = G
                elif i in s.off and child_of(mb) in src:
                    ds = F @ (src[child_of(mb)][:3, 3] - src[mb][:3, 3])
                    dt = Gt[:3, :3] @ s.off[i]
                    if np.linalg.norm(ds) > 1e-6 and np.linalg.norm(dt) > 1e-6:
                        G = Gt.copy(); G[:3, :3] = rot_between(dt, ds) @ Gt[:3, :3]; Gt = G
                LO[i] = np.linalg.inv(Gp) @ Gt
            Gn[i] = Gt
            for c in s.children.get(i, []): visit(c, Gt)
        for r in s.roots: visit(r, np.eye(4))
        s.G_last = Gn
        return LO
    def look_at(s, LO, target_g, weight=.7, bone='Bip01 Head', fwd=(0, 0, 1)):
        """Gira la cabeza hacia un punto (espacio glTF) con un peso dado; modifica LO."""
        i = s.name2node[bone]; G = s.G_last[i]; Gp = s.G_last[s.dst.parent[i]]
        f_now = G[:3, :3] @ s.G_rest[i][:3, :3].T @ np.asarray(fwd, float)
        d = np.asarray(target_g, float) - G[:3, 3]
        Q = rot_between(f_now, d)
        ang = np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1))
        if ang < 1e-5: return LO
        ax = np.array((Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1])); ax /= np.linalg.norm(ax)
        a = ang * weight; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Qw = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        Gn = G.copy(); Gn[:3, :3] = Qw @ G[:3, :3]
        LO[i] = np.linalg.inv(Gp) @ Gn
        return LO

def gltf_q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
