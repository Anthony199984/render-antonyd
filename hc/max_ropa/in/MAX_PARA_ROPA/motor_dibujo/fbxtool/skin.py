"""Personajes FBX con esqueleto (Mixamo): piel, huesos y animación por tiempo. Sin Blender.
Salida en Z-arriba, metros, mirando a -Y (convención del motor)."""
import io, sys, math, numpy as np
from PIL import Image
sys.path.insert(0, '/home/claude/fbxtool'); import fbx
TICK = 46186158000
C4 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])

def P70(n):
    out = {}
    for p in (fbx.find(n, 'Properties70') or [{'kids': []}])[0]['kids']:
        out[p['props'][0]] = p['props'][4:]
    return out

BIND_TM = False      # Mixamo (Swat): el Transform del cluster ya es el bind inverso del hueso
def bname(s): return s.split('\x00')[0].split(':')[-1]

def euler(deg, order=0):
    x, y, z = np.radians(deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    M = np.eye(4); M[:3, :3] = Rz @ Ry @ Rx; return M

def local_matrix(T, R, S, pre, post=(0, 0, 0)):
    M = np.eye(4); M[:3, 3] = T
    Sm = np.diag([S[0], S[1], S[2], 1.])
    return M @ euler(pre) @ euler(R) @ np.linalg.inv(euler(post)) @ Sm

def parse(path):
    top = fbx.read(path); Tn = {t['name']: t for t in top}; obj = Tn['Objects']; ids = {k['props'][0]: k for k in obj['kids']}
    par = {}; ch = {}
    for c in Tn['Connections']['kids']:
        a, b = c['props'][1], c['props'][2]; prop = c['props'][3] if len(c['props']) > 3 else None
        par.setdefault(a, []).append((b, prop)); ch.setdefault(b, []).append((a, prop))
    return ids, par, ch

class Character:
    def __init__(s, path, height=None):
        ids, par, ch = parse(path)
        # nodos Model (huesos y nulos)
        s.nodes = {}
        for i, n in ids.items():
            if n['name'] != 'Model': continue
            p = P70(n); nm = bname(n['props'][1])
            parent = [q for q, _ in par.get(i, []) if q in ids and ids[q]['name'] == 'Model']
            s.nodes[i] = dict(name=nm, parent=parent[0] if parent else None, T=np.array(p.get('Lcl Translation', [0, 0, 0]), float),
                              R=np.array(p.get('Lcl Rotation', [0, 0, 0]), float), S=np.array(p.get('Lcl Scaling', [1, 1, 1]), float),
                              pre=np.array(p.get('PreRotation', [0, 0, 0]), float), post=np.array(p.get('PostRotation', [0, 0, 0]), float), kind=n['props'][2])
        s.byname = {v['name']: k for k, v in s.nodes.items()}
        s.parts = []
        for g in [n for n in ids.values() if n['name'] == 'Geometry' and fbx.find(n, 'Vertices')]:
            gid = g['props'][0]; V = fbx.find(g, 'Vertices')[0]['props'][0].reshape(-1, 3).astype(float); pvi = fbx.find(g, 'PolygonVertexIndex')[0]['props'][0]
            # lector tolerante: normales y UV pueden venir Direct o IndexToDirect, por vértice o por esquina (Blender re-exporta así)
            nl = fbx.find(g, 'LayerElementNormal')[0]; N = fbx.find(nl, 'Normals')[0]['props'][0].reshape(-1, 3).astype(float); nmap = fbx.find(nl, 'MappingInformationType')[0]['props'][0]
            nidx = fbx.find(nl, 'NormalsIndex'); NI = nidx[0]['props'][0].astype(int) if nidx and fbx.find(nl, 'ReferenceInformationType')[0]['props'][0] == 'IndexToDirect' else None
            if NI is not None: N = N[NI]
            ul = fbx.find(g, 'LayerElementUV')[0]; UV = fbx.find(ul, 'UV')[0]['props'][0].reshape(-1, 2); umap = fbx.find(ul, 'MappingInformationType')[0]['props'][0]
            uix = fbx.find(ul, 'UVIndex'); UI = uix[0]['props'][0].astype(int) if uix and fbx.find(ul, 'ReferenceInformationType')[0]['props'][0] == 'IndexToDirect' else np.arange(len(UV))
            if umap != 'ByPolygonVertex': UI = None
            # material por polígono (varios materiales en una sola malla, p. ej. Mixamo cuerpo + cabeza)
            ml = fbx.find(g, 'LayerElementMaterial'); MP = None
            if ml and fbx.find(ml[0], 'Materials'):
                MP = np.asarray(fbx.find(ml[0], 'Materials')[0]['props'][0]).astype(int).ravel()
                if fbx.find(ml[0], 'MappingInformationType')[0]['props'][0] == 'AllSame': MP = None
            cv, cuv, cn, cm = [], [], [], []; poly = []; npoly = 0
            for i, ix in enumerate(pvi):
                poly.append((ix if ix >= 0 else -ix - 1, i))
                if ix < 0:
                    mi = int(MP[npoly]) if MP is not None and npoly < len(MP) else 0
                    for j in range(1, len(poly) - 1):
                        for (a, c) in (poly[0], poly[j], poly[j + 1]):
                            cv.append(a); cuv.append(UV[UI[c]] if UI is not None else UV[a]); cn.append(N[c] if nmap == 'ByPolygonVertex' else N[a]); cm.append(mi)
                    poly = []; npoly += 1
            cv = np.array(cv); cuv = np.array(cuv, float); cn = np.array(cn); cm = np.array(cm)
            model = [q for q, _ in par.get(gid, []) if q in ids and ids[q]['name'] == 'Model'][0]
            # piel
            clusters = []
            for sk, _ in ch.get(gid, []):
                if sk not in ids or ids[sk]['name'] != 'Deformer': continue
                for cl, _ in ch.get(sk, []):
                    c = ids.get(cl)
                    if not c or c['name'] != 'Deformer' or not fbx.find(c, 'Indexes'): continue
                    bone = [q for q, _ in ch.get(cl, []) if q in ids and ids[q]['name'] == 'Model'][0]
                    Tm = fbx.find(c, 'Transform')[0]['props'][0].reshape(4, 4).T; TL = fbx.find(c, 'TransformLink')[0]['props'][0].reshape(4, 4).T
                    clusters.append(dict(Tm=Tm, TL=TL, bone=bone, idx=fbx.find(c, 'Indexes')[0]['props'][0].astype(int), w=fbx.find(c, 'Weights')[0]['props'][0].astype(float), bind=(Tm if (np.allclose(Tm @ TL, np.eye(4), atol=1e-3) or BIND_TM) else np.linalg.inv(TL) @ Tm)))
            # textura difusa / opacidad
            diff = opac = None; mats = []; matn = []
            for m, _ in ch.get(model, []):
                if m not in ids or ids[m]['name'] != 'Material': continue
                mats.append(None); matn.append(bname(ids[m]['props'][1]) if len(ids[m]['props']) > 1 else 'm')
                for tx, prop in ch.get(m, []):
                    if tx not in ids or ids[tx]['name'] != 'Texture': continue
                    for vd, _ in ch.get(tx, []):
                        if vd not in ids or ids[vd]['name'] != 'Video': continue
                        cont = fbx.find(ids[vd], 'Content'); data = cont[0]['props'][0] if cont and cont[0]['props'] else b''
                        if len(data) < 100: continue
                        im = Image.open(io.BytesIO(bytes(data)))
                        if prop and 'Diffuse' in prop and mats[-1] is None: mats[-1] = im.convert('RGB')
                        if prop and 'Diffuse' in prop and diff is None: diff = im.convert('RGB')
                        if prop and ('Transparen' in prop or 'Opacity' in prop): opac = im.convert('L')
            W = np.zeros((len(V), len(clusters)))
            for k, c in enumerate(clusters): W[c['idx'], k] = c['w']
            tot = W.sum(1); free = tot < 1e-6
            s.parts.append(dict(name=bname(g['props'][1]) or 'part', V=V, cv=cv, uv=cuv, cn=cn, clusters=clusters, W=W / np.maximum(tot, 1e-9)[:, None], free=free, model=model, diff=diff, opac=opac, cm=cm, mats=mats, mat_names=matn))
        s.scale = 1.0; s.height = None
        bind = s.skin(None, apply_scale=False)
        if bind: zs = np.concatenate([b[0][:, 2] for b in bind])
        else:   # FBX sin piel (solo esqueleto): la altura sale de las articulaciones
            Gb = s.globals(); zs = np.array([(C4[:3, :3] @ g[:3, 3])[2] for g in Gb.values()])
        s.zmin = zs.min(); h = zs.max() - zs.min()
        s.scale = (height / h) if height else .01

    def globals(s, anim=None, t=0.):
        G = {}
        def g(i):
            if i in G: return G[i]
            n = s.nodes[i]; T, R, S = n['T'], n['R'], n['S']
            if anim is not None:
                T, R, S = anim.sample(n['name'], t, T, R, S)
            M = local_matrix(T, R, S, n['pre'], n['post'])
            G[i] = (g(n['parent']) @ M) if n['parent'] in s.nodes else M
            return G[i]
        for i in s.nodes: g(i)
        return G

    def skin(s, anim=None, t=0., apply_scale=True, root=None):
        G = s.globals(anim, t) if anim is not None else None
        out = []
        for p in s.parts:
            if p['clusters']:
                Ks = np.stack([(G[c['bone']] if G is not None else s.globals()[c['bone']]) @ c['bind'] for c in p['clusters']]) if G is not None else None
                if Ks is None:
                    Gb = s._bindG if hasattr(s, '_bindG') else None
                    if Gb is None: s._bindG = s.globals(); Gb = s._bindG
                    Ks = np.stack([Gb[c['bone']] @ c['bind'] for c in p['clusters']])
                Mv = (p['W'] @ Ks.reshape(len(Ks), 16)).reshape(-1, 4, 4)
                if p['free'].any():
                    Mm = (G or s.globals())[p['model']]; Mv[p['free']] = Mm
            else:
                Mm = (G or s.globals())[p['model']]; Mv = np.broadcast_to(Mm, (len(p['V']), 4, 4))
            Vh = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['V']) + Mv[:, :3, 3]
            Mc = Mv[p['cv']]; Nc = np.einsum('vij,vj->vi', Mc[:, :3, :3], p['cn'])
            P = Vh[p['cv']]
            P = P @ C4[:3, :3].T; Nc = Nc @ C4[:3, :3].T
            if apply_scale: P = (P - np.array([0, 0, s.zmin])) * s.scale
            Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-9)
            out.append((P, Nc))
        return out

class Anim:
    def __init__(s, path):
        ids, par, ch = parse(path); s.curves = {}; s.length = 0
        for i, n in ids.items():
            if n['name'] != 'AnimationCurveNode': continue
            targ = [(q, prop) for q, prop in par.get(i, []) if q in ids and ids[q]['name'] == 'Model']
            if not targ: continue
            bone = bname(ids[targ[0][0]]['props'][1]); prop = targ[0][1]; key = {'Lcl Translation': 'T', 'Lcl Rotation': 'R', 'Lcl Scaling': 'S'}.get(prop)
            if not key: continue
            axes = {}
            for c, cp in ch.get(i, []):
                if c not in ids or ids[c]['name'] != 'AnimationCurve': continue
                kt = fbx.find(ids[c], 'KeyTime')[0]['props'][0] / TICK; kv = fbx.find(ids[c], 'KeyValueFloat')[0]['props'][0].astype(float)
                axes['XYZ'.index(cp[-1])] = (kt, kv); s.length = max(s.length, kt[-1])
            s.curves.setdefault(bone, {})[key] = axes
    def sample(s, bone, t, T, R, S, loop=True):
        c = s.curves.get(bone)
        if not c: return T, R, S
        if loop and s.length > 0: t = t % s.length
        def ev(k, base):
            if k not in c: return base
            out = np.array(base, float)
            for ax, (kt, kv) in c[k].items(): out[ax] = np.interp(t, kt, kv)
            return out
        return ev('T', T), ev('R', R), ev('S', S)

def interleave(P, N, UV):
    a = np.empty((len(P), 11), np.float32); a[:, 0:3] = P; a[:, 3:6] = N; a[:, 6:8] = UV; a[:, 8:11] = 1; return a

UPPER_KEYS = ('Spine', 'Neck', 'Head', 'Shoulder', 'Arm', 'Hand')
def is_upper(b): return any(k in b for k in UPPER_KEYS) and 'UpLeg' not in b

class Comp:
    """Animación compuesta por huesos: capas (anim, f_tiempo, filtro) + giros extra por hueso (grados).
    Para ajustar posturas: jinete con piernas abiertas, celebración sentado con brazos de pie, etc."""
    def __init__(s, layers, offsets=None, root_xy_zero=False):
        s.layers = layers; s.offsets = offsets or {}; s.length = max(getattr(a, 'length', 0) for a, _, _ in layers); s.rz = root_xy_zero
    def sample(s, bone, t, T, R, S, loop=True):
        out = (T, R, S)
        for a, f, filt in s.layers:
            if filt is None or filt(bone): out = a.sample(bone, f(t), T, R, S)
        T, R, S = out
        if s.rz and bone == 'Hips': T = np.array(T, float); T[0] = 0; T[2] = 0
        if bone in s.offsets:
            o = s.offsets[bone]; o = o(t) if callable(o) else o; R = np.array(R, float) + np.asarray(o, float)
        return T, R, S

def euler_inv(M):
    """Inversa de euler(): M = Rz@Ry@Rx → grados (x, y, z)."""
    R = M[:3, :3]; y = math.asin(-max(-1, min(1, R[2, 0])))
    if abs(R[2, 0]) < .9999: x = math.atan2(R[2, 1], R[2, 2]); z = math.atan2(R[1, 0], R[0, 0])
    else: x = math.atan2(-R[1, 2], R[1, 1]); z = 0
    return np.degrees([x, y, z])

class AnimPre(Anim):
    """Animación Mixamo (con PreRotation en los huesos) para un personaje re-exportado desde Blender (PreRotation horneada en
    Lcl Rotation): devuelve R = pre ∘ R_anim."""
    def __init__(s, path):
        super().__init__(path); ids, par, ch = parse(path); s.pre = {}
        for i, n in ids.items():
            if n['name'] == 'Model':
                p = P70(n); s.pre[bname(n['props'][1])] = np.array(p.get('PreRotation', [0, 0, 0]), float)
    def sample(s, bone, t, T, R, S, loop=True):
        T2, R2, S2 = super().sample(bone, t, T, R, S, loop)
        if bone in s.curves and 'R' in s.curves[bone] and bone in s.pre and np.any(s.pre[bone]):
            R2 = euler_inv(euler(s.pre[bone]) @ euler(R2))
        return T2, R2, S2
