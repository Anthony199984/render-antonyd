"""Max sobre el esqueleto SOMA77 de Kimodo, SIN retarget.

Cómo funciona (exacto, sin igualar direcciones cuadro a cuadro):
- Las articulaciones SOMA se colocan dentro del cuerpo de Max (en las suyas mixamorig y, donde SOMA tiene
  huesos extra, en puntos proporcionales: metacarpos, Neck2, puntas).
- Max está en pose A y Kimodo en su propio reposo. Para cada hueso se calcula UNA vez el giro A_j que lleva
  el reposo del BVH a la pose de Max; ese giro es el bind del hueso. Con eso, el giro global de Kimodo se
  usa tal cual: piel_j = [G_kimodo_j | p_j] · inv([A_j | pA_j]).
- Los largos de hueso son los de Max (no se estira nada) y la cadera se mueve escalada por la altura de cadera.
- Pesos: los de max_v1 pasados por nombre (tabla PESOS). Neck se reparte entre Neck1 y Neck2 por altura.

Uso:
    from soma import MaxSoma, Clip
    m = MaxSoma()                                  # carga Personajes/Max/max_v1.glb
    c = Clip('caminar.bvh')                        # BVH tal cual sale de Kimodo
    res = m.pose(c, t)                             # mismo formato que gltf.Skinned.pose: [(V, N) por primitiva], Z arriba
    m.joint_pos('LeftHand')                        # posición (Z arriba, metros) tras el último pose
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.environ.get('RAIZ', '/home/claude'), 'fbxtool'))
import gltf
from .bvh import Clip

RAIZ = os.environ.get('RAIZ', '/home/claude')
MAX_GLB = os.path.join(RAIZ, 'Personajes/Max/max_v1.glb')
TPOSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'somaskel77_standard_tpose.bvh')
MX = 'mixamorig:'
DEDOS = ('Index', 'Middle', 'Ring', 'Pinky')

def _pesos():
    t = {'Hips': 'Hips', 'Spine': 'Spine1', 'Spine1': 'Spine2', 'Spine2': 'Chest', 'Neck': 'Neck1', 'Head': 'Head'}
    for L in ('Left', 'Right'):
        for a, b in (('Shoulder', 'Shoulder'), ('Arm', 'Arm'), ('ForeArm', 'ForeArm'), ('Hand', 'Hand'),
                     ('UpLeg', 'Leg'), ('Leg', 'Shin'), ('Foot', 'Foot'), ('ToeBase', 'ToeBase')):
            t[L + a] = L + b
        for k in (1, 2, 3):
            t[f'{L}HandThumb{k}'] = f'{L}HandThumb{k}'
            for d in DEDOS: t[f'{L}Hand{d}{k}'] = f'{L}Hand{d}{k + 1}'   # SOMA Dedo1 = metacarpo
    return t
PESOS = _pesos()

def _norm(v): n = np.linalg.norm(v); return v/n if n > 1e-12 else v
def _swing(a, b):
    """Giro mínimo que lleva la dirección a a la dirección b."""
    a, b = _norm(a), _norm(b); v = np.cross(a, b); c = float(np.dot(a, b)); s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0: return np.eye(3)
        p = _norm(np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0])); return 2*np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K*((1 - c)/s**2)
def _base(v1, v2):
    e1 = _norm(v1); e2 = _norm(v2 - e1*np.dot(v2, e1)); return np.stack([e1, e2, np.cross(e1, e2)], 1)
def _M(R, p): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M

class MaxSoma:
    CADERA_PANTALON = 0.0      # 0 = pesos del pantalón como en max_v1 (probado: pasar peso a la cadera abre la cintura)
    def __init__(s, glb=MAX_GLB, altura=1.874, esqueleto=TPOSE):
        s.base = gltf.Skinned(glb, altura); s.prims = s.base.prims; s.J = s.base.J
        sk = Clip(esqueleto); s.nombres = sk.nombres; s.padres = sk.padres; s.idx = sk.idx
        nodos = s.J['nodes']
        s.bind_mx = {nodos[j]['name'].replace(MX, ''): np.linalg.inv(s.base.ibm[k])[:3, 3] for k, j in enumerate(s.base.joints)}
        s.pA = s._ajustar(sk)
        s._pesos()
        s._ocultar_piel_tapada()
        s._cache = {}
        s.G = None; s.Wm = None

    # ---------------------------------------------------------------- articulaciones dentro de Max
    def _ajustar(s, sk):
        m = s.bind_mx; P = {}
        k = 1.874/sk.reposo[:, 1].max()
        P['Root'] = np.array([m['Hips'][0], 0., m['Hips'][2]])
        # En max_v1 el hueso Hips->Spine mide 0,3 mm: Spine1 de SOMA se pone a su altura proporcional (entre Hips y Spine1 de Max)
        r = sk.reposo; fs = np.linalg.norm(r[sk.idx['Spine1']] - r[sk.idx['Hips']])/np.linalg.norm(r[sk.idx['Spine2']] - r[sk.idx['Hips']])
        P['Hips'] = m['Hips']; P['Spine1'] = m['Hips'] + fs*(m['Spine1'] - m['Hips']); P['Spine2'] = m['Spine1']; P['Chest'] = m['Spine2']
        P['Neck1'] = m['Neck']; P['Neck2'] = (m['Neck'] + m['Head'])/2; P['Head'] = m['Head']
        cab = [p for p in s.prims if 'Head' in p['mname']]
        top = max(float(p['P'][:, 1].max()) for p in cab) if cab else m['Head'][1] + .2
        P['HeadEnd'] = np.array([m['Head'][0], top, m['Head'][2]])
        for n in ('Jaw', 'LeftEye', 'RightEye'): P[n] = m['Head'] + k*(sk.reposo[sk.idx[n]] - sk.reposo[sk.idx['Head']])
        for L in ('Left', 'Right'):
            for a in ('Shoulder', 'Arm', 'ForeArm', 'Hand'): P[L + a] = m[L + a]
            h = m[L + 'Hand']
            P[f'{L}HandThumb1'], P[f'{L}HandThumb2'], P[f'{L}HandThumb3'] = (m[f'{L}HandThumb{i}'] for i in (1, 2, 3))
            P[f'{L}HandThumbEnd'] = m[f'{L}HandThumb3'] + 1.0*(m[f'{L}HandThumb3'] - m[f'{L}HandThumb2'])
            for d in DEDOS:
                r = sk.reposo; i0, i1, i2 = (sk.idx[f'{L}Hand'], sk.idx[f'{L}Hand{d}1'], sk.idx[f'{L}Hand{d}2'])
                f = np.linalg.norm(r[i1] - r[i0])/np.linalg.norm(r[i2] - r[i0])          # dónde cae el metacarpo en SOMA
                P[f'{L}Hand{d}1'] = h + f*(m[f'{L}Hand{d}1'] - h)
                for kk in (1, 2, 3): P[f'{L}Hand{d}{kk + 1}'] = m[f'{L}Hand{d}{kk}']
                P[f'{L}Hand{d}End'] = m[f'{L}Hand{d}3'] + .85*(m[f'{L}Hand{d}3'] - m[f'{L}Hand{d}2'])
            P[L + 'Leg'] = m[L + 'UpLeg']; P[L + 'Shin'] = m[L + 'Leg']; P[L + 'Foot'] = m[L + 'Foot']; P[L + 'ToeBase'] = m[L + 'ToeBase']
            # punta del pie: con la MISMA inclinación respecto al pie que en SOMA (antes iba horizontal y SOMA la tiene 14° abajo:
            # el bind giraba la puntera ~10° y el tenis se veía doblado hacia abajo a mitad del pie, incluso en reposo)
            r = sk.reposo; ix = sk.idx
            vs = r[ix[L + 'ToeBase']] - r[ix[L + 'Foot']]; vm = m[L + 'ToeBase'] - m[L + 'Foot']
            ts = r[ix[L + 'ToeEnd']] - r[ix[L + 'ToeBase']]
            P[L + 'ToeEnd'] = m[L + 'ToeBase'] + .065*_norm(_swing(vs, vm) @ ts)
        falta = [n for n in s.nombres if n not in P]
        assert not falta, falta
        return P

    # ---------------------------------------------------------------- pesos
    def _pesos(s):
        nodos = s.J['nodes']; mx_a_soma = []
        for j in s.base.joints:
            n = nodos[j]['name'].replace(MX, ''); mx_a_soma.append(s.idx[PESOS[n]] if n in PESOS else s.idx['Hips'])
        mx_a_soma = np.array(mx_a_soma); i_neck = list(PESOS).index('Neck')
        k_neck = [k for k, j in enumerate(s.base.joints) if nodos[j]['name'] == MX + 'Neck'][0]
        y2 = s.pA['Neck2'][1]
        s.Jn, s.Wn = [], []
        piernas = (s.idx['LeftLeg'], s.idx['RightLeg'])
        for p in s.prims:
            Jn = mx_a_soma[p['J']]; W = p['W'].copy()
            arriba = (p['P'][:, 1] > y2)[:, None] & (p['J'] == k_neck)
            Jn = np.where(arriba, s.idx['Neck2'], Jn)
            if p['mname'] == 'Pants' and MaxSoma.CADERA_PANTALON > 0:
                # el pantalón de max_v1 va pesado solo a los muslos y la cadera de Max queda sobre la cintura: al llevar la pierna
                # atrás, los glúteos giraban con el muslo y el pantalón se doblaba. Parte del peso alto pasa a la cadera (degradado).
                y = p['P'][:, 1]; y0 = s.pA['LeftShin'][1]*.35 + s.pA['LeftLeg'][1]*.65 - .10; y1 = y.max()
                u = np.clip((y - y0)/(y1 - y0), 0, 1); f = MaxSoma.CADERA_PANTALON*u*u*(3 - 2*u)
                enp = np.isin(Jn, piernas); quita = (W*enp)*f[:, None]
                W = W - quita; Jn = np.c_[Jn, np.full(len(Jn), s.idx['Hips'])]; W = np.c_[W, quita.sum(1)]
            s.Jn.append(Jn); s.Wn.append(W)

    def _ocultar_piel_tapada(s):
        """La piel bajo el pantalón y los tenis no se ve desde fuera; si se dibuja, asoma por el pantalón al subir mucho la pierna
        (el pantalón va pesado al muslo y la piel de debajo, en parte a la cadera). Se quitan esos triángulos de piel."""
        pants = [p for p in s.prims if p['mname'] == 'Pants']
        if not pants: return
        cintura = float(pants[0]['P'][:, 1].max()) - .03
        for p in s.prims:
            if not p['mname'].startswith(('Std_Skin_Body', 'Std_Skin_Leg')): continue
            if 'idx_orig' not in p: p['idx_orig'] = p['idx']
            tri = p['idx_orig'].reshape(-1, 3); y = p['P'][tri, 1]
            p['idx'] = tri[~(y < cintura).all(1)].ravel()

    # ---------------------------------------------------------------- bind según el reposo del BVH
    def _bind(s, clip):
        clave = (clip.reposo.round(5).tobytes(), tuple(clip.nombres))
        if clave in s._cache: return s._cache[clave]
        r = np.array([clip.reposo[clip.idx[n]] for n in s.nombres]); pA = s.pA; ix = s.idx
        hijo = {}
        for n in s.nombres:
            if n in ('Root', 'Hips', 'Chest', 'LeftHand', 'RightHand') or n.endswith('End') or n in ('Jaw', 'LeftEye', 'RightEye'): continue
            cand = {'Neck1': 'Neck2', 'Neck2': 'Head', 'Head': 'HeadEnd', 'LeftShoulder': 'LeftArm', 'RightShoulder': 'RightArm'}
            if n in cand: hijo[n] = cand[n]; continue
            hs = [c for c, p in enumerate(s.padres) if p == ix[n]]
            if hs: hijo[n] = s.nombres[hs[0]]
        A = {}
        for j, n in enumerate(s.nombres):
            p = s.padres[j]; Ap = A[s.nombres[p]] if p >= 0 else np.eye(3)
            if n == 'Root': A[n] = np.eye(3)
            elif n in ('Hips', 'Chest'):
                arr = 'Spine2' if n == 'Hips' else 'Neck1'; lado = ('LeftLeg', 'RightLeg') if n == 'Hips' else ('LeftShoulder', 'RightShoulder')
                Br = _base(r[ix[arr]] - r[j], r[ix[lado[0]]] - r[ix[lado[1]]]); Ba = _base(pA[arr] - pA[n], pA[lado[0]] - pA[lado[1]])
                A[n] = Ba @ Br.T
            elif n in ('LeftHand', 'RightHand'):
                L = n[:-4]
                Br = _base(r[ix[L + 'HandMiddle2']] - r[j], r[ix[L + 'HandIndex2']] - r[ix[L + 'HandPinky2']])
                Ba = _base(pA[L + 'HandMiddle2'] - pA[n], pA[L + 'HandIndex2'] - pA[L + 'HandPinky2'])
                A[n] = Ba @ Br.T
            elif n in hijo:
                c = hijo[n]; A[n] = _swing(Ap @ (r[ix[c]] - r[j]), pA[c] - pA[n]) @ Ap
            else: A[n] = Ap
        Al = np.stack([A[n] for n in s.nombres]); Pl = np.stack([pA[n] for n in s.nombres])
        off = np.zeros_like(Pl)
        for j, p in enumerate(s.padres):
            if p >= 0: off[j] = Al[p].T @ (Pl[j] - Pl[p])
        ibm = np.stack([np.linalg.inv(_M(Al[j], Pl[j])) for j in range(len(Al))])
        def pierna(P, ix): return sum(np.linalg.norm(P[ix[a]] - P[ix[c]]) for a, c in (('LeftLeg', 'LeftShin'), ('LeftShin', 'LeftFoot')))
        k = pierna(Pl, ix)/pierna(clip.reposo, clip.idx)          # el avance se escala por el largo de pierna (no por la cadera)
        b = dict(A=Al, P=Pl, off=off, ibm=ibm, k=k, hips0=clip.reposo[clip.idx['Hips']])
        s._cache[clave] = b; return b

    # ---------------------------------------------------------------- bloqueo de pies
    def _contactos(s, clip, vmax=.12, hmax=.06):
        """Por pie: peso de bloqueo por cuadro (0..1) y cuadro donde empieza cada apoyo, detectados en el esqueleto de Kimodo."""
        if getattr(clip, '_contactos', None) is not None: return clip._contactos
        out = {}
        for L in ('Left', 'Right'):
            p = clip.P[:, clip.idx[L + 'Foot']]; v = np.r_[0, np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)*clip.fps]
            h = p[:, 1] - np.percentile(p[:, 1], 3)
            c = (v < vmax) & (h < hmax)
            for _ in range(2): c = c | (np.r_[c[1:], False] & np.r_[False, c[:-1]])    # tapa huecos de 1 cuadro
            ini = np.full(clip.n, -1); w = np.zeros(clip.n); f = 0
            while f < clip.n:
                if not c[f]: f += 1; continue
                g = f
                while g + 1 < clip.n and c[g + 1]: g += 1
                if g - f >= 3:
                    for q in range(f, g + 1):
                        ini[q] = f; w[q] = min(1.0, (q - f + 1)/2.0, (g - q + 1)/3.0)
                f = g + 1
            out[L] = (w, ini)
        clip._contactos = out; return out

    def _ik_pierna(s, G, Pm, b, L, objetivo):
        ix = s.idx; iH, iK, iA = ix[L + 'Leg'], ix[L + 'Shin'], ix[L + 'Foot']
        H, K, A = Pm[iH], Pm[iK], Pm[iA]; l1, l2 = np.linalg.norm(K - H), np.linalg.norm(A - K)
        d_v = objetivo - H; d = min(np.linalg.norm(d_v), (l1 + l2)*.9995); e = _norm(d_v)
        x = (l1*l1 - l2*l2 + d*d)/(2*d); hh = np.sqrt(max(l1*l1 - x*x, 0.0))
        pol = (K - H) - e*np.dot(K - H, e); pol = _norm(pol) if np.linalg.norm(pol) > 1e-6 else _norm(np.cross(e, [1, 0, 0]))
        K2 = H + x*e + hh*pol; A2 = H + d*e
        R1 = _swing(K - H, K2 - H); G[iH] = R1 @ G[iH]
        Gs = R1 @ G[iK]; A1 = K2 + Gs @ b['off'][iA]; G[iK] = _swing(A1 - K2, A2 - K2) @ Gs
        for j in (iK, iA, ix[L + 'ToeBase'], ix[L + 'ToeEnd']): Pm[j] = Pm[s.padres[j]] + G[s.padres[j]] @ b['off'][j]

    # ---------------------------------------------------------------- pose
    def _sin_bloqueo(s, clip, t, bucle=False):
        b = s._bind(clip); G, P = clip.globales(t, bucle)
        orden = [clip.idx[n] for n in s.nombres]; G = G[orden].copy(); Ph = P[clip.idx['Hips']]
        Pm = np.zeros((len(s.nombres), 3)); ih = s.idx['Hips']
        for j, p in enumerate(s.padres):
            if j == ih: Pm[j] = b['P'][ih] + b['k']*(Ph - b['hips0'])
            elif p < 0: Pm[j] = b['P'][j]
            else: Pm[j] = Pm[p] + G[p] @ b['off'][j]
        return G, Pm, b

    def globales(s, clip, t, bucle=False, pies=True):
        """Matrices globales de cada hueso SOMA de Max (espacio glTF, Y arriba) en el tiempo t.
        pies=True: el pie apoyado se queda clavado donde aterrizó (IK de dos huesos; el giro del pie es el de Kimodo)."""
        G, Pm, b = s._sin_bloqueo(clip, t, bucle)
        if not pies: return G, Pm, b
        f = t*clip.fps
        if bucle and clip.n > 1: f %= (clip.n - 1)
        f = int(round(min(max(f, 0), clip.n - 1)))
        for L, (w, ini) in s._contactos(clip).items():
            if w[f] <= 0: continue
            obj = s._sin_bloqueo(clip, ini[f]/clip.fps)[1][s.idx[L + 'Foot']]
            obj = Pm[s.idx[L + 'Foot']]*(1 - w[f]) + obj*w[f]
            s._ik_pierna(G, Pm, b, L, obj)
        return G, Pm, b

    def pose_mats(s, G, Pm, b, apply=True):
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        s.G, s.Wp = G, Pm
        res = []; Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        for p, Jn, Wn in zip(s.prims, s.Jn, s.Wn):
            Mv = np.einsum('vk,vkij->vij', Wn, K[Jn])
            V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p.get('Pm', p['P'])) + Mv[:, :3, 3]
            Nn = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N'])
            V = V @ Cz.T; Nn = Nn @ Cz.T
            if apply: V = (V - np.array([0, 0, s.base.zoff]))*s.base.scale
            Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            res.append((V[p['idx']], Nn[p['idx']]))
        return res

    def pose(s, clip, t, bucle=False, apply=True):
        G, Pm, b = s.globales(clip, t, bucle); return s.pose_mats(G, Pm, b, apply)

    def set_morph(s, weights): s.base.set_morph(weights)

    def joint_pos(s, nombre):
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        return ((Cz @ s.Wp[s.idx[nombre]]) - [0, 0, s.base.zoff])*s.base.scale
