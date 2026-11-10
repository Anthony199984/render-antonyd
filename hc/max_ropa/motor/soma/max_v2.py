"""Max v2 en el motor (archivo NUEVO: no cambia nada de lo que ya había; MaxSoma sigue igual para max_v1).

    from soma.max_v2 import MaxV2
    max = MaxV2(glb="Personajes/Max/max_v2_poloche.glb")                  # con poloche
    max = MaxV2(glb="Personajes/Max/max_v2_poloche.glb", poloche=False)   # sin camisa (se quita al cargar)
    res = max.pose(clip, t)                                                # igual que MaxSoma
    # para volver a la forma de MaxSoma en un hueso (1 = como MaxSoma): MaxV2.IGUALAR['LeftShoulder'] = 1.0

Qué cambia respecto a MaxSoma:
 0. Postura: hombros y cuello. El esqueleto de Max (mixamorig) pone la clavícula 28° hacia abajo y la cabeza detrás del
    cuello; SOMA (Kimodo) los tiene horizontal y hacia adelante. MaxSoma iguala esas direcciones hueso por hueso y por eso,
    al animar, los hombros se veían encogidos hacia arriba y el cuello doblado hacia adelante. MaxV2 toma la postura
    natural de Max a medias en el cuello y las clavículas (tabla IGUALAR): hombros relajados y cuello recto, y Kimodo
    sigue mandando el movimiento y la mirada.
Y solo si el glb trae una prenda "Poloche_*":
 1. Piel tapada: la piel del torso y de los hombros que queda bien adentro del poloche no se dibuja (como ya se hace con la
    piel bajo el pantalón). Así la piel nunca atraviesa la tela al agacharse, estirarse o levantar los brazos. La piel cerca
    del cuello y de las mangas sí se dibuja (ahí se ve).
 2. Física del borde: la parte de abajo del poloche (desde unos 18 cm sobre el bajo) se mueve con resorte y amortiguación:
    se retrasa un poco y se balancea al caminar, girar o agacharse; el resto del poloche va pegado al cuerpo.
 3. Choque con las piernas: el borde nunca queda por dentro del muslo (al subir la pierna, el muslo empuja la tela hacia
    afuera en vez de atravesarla). Cápsulas cadera-rodilla del grosor del pantalón; lo que en reposo ya está más cerca no se empuja.
La física necesita los cuadros en orden (t creciente). Si t salta hacia atrás o más de 0,25 s, la tela se reinicia quieta.
"""
import numpy as np

from .rig import MaxSoma, _M, _swing, _base

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def _vecino(ref, q):
    """Índice y distancia del punto de ref más cercano a cada q."""
    if cKDTree is not None:
        d, i = cKDTree(ref).query(q)
        return i, d
    i = np.empty(len(q), int); d = np.empty(len(q))
    for a in range(0, len(q), 512):
        dd = ((q[a:a + 512, None, :] - ref[None]) ** 2).sum(-1)
        i[a:a + 512] = dd.argmin(1); d[a:a + 512] = np.sqrt(dd.min(1))
    return i, d


def _fraccion(R, f):
    """Giro R hecho solo en la fracción f (mismo eje)."""
    if f >= 1: return R
    if f <= 0: return np.eye(3)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); ang = np.arccos(c)
    if ang < 1e-7: return np.eye(3)
    eje = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    K = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]]); a = ang * f
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _rodrigues(rv):
    a = np.linalg.norm(rv)
    if a < 1e-9: return np.eye(3)
    k = rv / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _norm_(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


class MaxV2(MaxSoma):
    # cuánto se iguala cada hueso al reposo de Kimodo (1 = como MaxSoma, 0 = se queda la postura natural de Max)
    # (probado con caminar, agacharse y celebrar: la columna igualada del todo, el cuello a medias y la cabeza igualada
    # dejan el cuello recto con la mirada al frente; las clavículas a medias dejan los hombros relajados)
    IGUALAR = {}
    # AGARRES: las manos quedan EXACTO donde las pone MaxSoma (donde agarres_ik las calculó). Los hombros y el cuello
    # siguen con la postura nueva; solo el brazo (hombro->codo->muñeca) se reacomoda con IK de dos huesos para llegar
    # al mismo punto, con el codo hacia el mismo lado y el giro de la mano y los dedos sin cambiar.
    MANOS_COMO_ANTES = True
    # POSTURA NEUTRA de Kimodo (giro local promedio de cada hueso al caminar, simétrico): cuando Kimodo está en esa
    # postura normal, la columna, el cuello, la cabeza y los hombros de Max quedan EXACTO como en su modelo (de pie,
    # hombros relajados, cuello recto). Kimodo solo aporta lo que se aparta de esa postura (agacharse, girar, mirar...).
    # Sin esto, el esqueleto de Max (clavícula 28° abajo, pecho 16° adelante) se igualaba al de Kimodo y salían los
    # hombros encogidos, el pecho salido y la espalda hundida.
    NEUTRO = {'Spine1': (-0.011, 0.0, 0.0), 'Spine2': (-0.090, 0.0, 0.0), 'Chest': (0.257, 0.0, 0.0),
              'Neck1': (0.099, 0.0, 0.0), 'Neck2': (-0.062, 0.0, 0.0), 'Head': (-0.119, 0.0, 0.0),
              'LeftShoulder': (0.253, -0.212, -0.284), 'RightShoulder': (0.253, 0.212, 0.284)}
    FISICA = False           # balanceo del borde de abajo (False: la camiseta va pegada al cuerpo; el choque con las piernas sigue)
    BORDE_ALTO = 1.065       # (m, glTF del bind) desde aquí hacia abajo la tela se mueve con física
    BORDE_BAJO = 0.885       # alto del bajo del poloche
    FRECUENCIA = 2.3         # Hz del resorte (más bajo = tela más suelta)
    AMORTIGUA = 0.42         # 0..1 (más alto = se calma antes)
    DESVIO_MAX = 0.05        # m: lo más que se aparta del cuerpo
    MARGEN_AFUERA = 0.011    # m: separación mínima entre la cara de afuera y el pantalón
    MARGEN_ADENTRO = 0.0065  # m: lo mismo para la cara de adentro (mantiene el grosor de la tela)
    PIEL_ADENTRO = 0.004     # m: piel más adentro que esto de la tela...
    PIEL_ABERTURA = 0.12     # m: ...y a más de esto del cuello / mangas / bajo, no se dibuja (cerca de las aberturas sí, para que no se vea hueco)

    def __init__(s, *a, poloche=True, camiseta=None, **k):
        super().__init__(*a, **k)
        if camiseta is not None: poloche = camiseta
        if not poloche:      # el mismo glb sin la prenda: se quita al cargar (el archivo no cambia)
            quedan = [i for i, p in enumerate(s.prims) if not p['mname'].startswith('Poloche')]
            s.prims[:] = [s.prims[i] for i in quedan]
            s.Jn = [s.Jn[i] for i in quedan]; s.Wn = [s.Wn[i] for i in quedan]
        s.ropa = [i for i, p in enumerate(s.prims) if p['mname'].startswith('Poloche')]
        s.pantalon = [i for i, p in enumerate(s.prims) if p['mname'] == 'Pants']
        s._estado = None
        s._t = None; s._clip = None
        s._f = {}; s._choque = {}
        s._grupo = {}; s._ady = {}
        for i in s.ropa:
            P0 = s.prims[i]['P']
            _, g = np.unique(np.round(P0, 6), axis=0, return_inverse=True); g = g.ravel()
            F = g[s.prims[i]['idx'].reshape(-1, 3)]; n = g.max() + 1
            try:
                from scipy.sparse import coo_matrix
                A = coo_matrix((np.ones(len(F) * 6), (np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]],
                                                     np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])), shape=(n, n)).tocsr()
                A.data[:] = 1
                s._ady[i] = (A, np.maximum(np.asarray(A.sum(1)).ravel(), 1))
            except ImportError:
                s._ady[i] = None
            s._grupo[i] = g
        for i in s.ropa:
            y = s.prims[i]['P'][:, 1]
            u = np.clip((s.BORDE_ALTO - y) / (s.BORDE_ALTO - s.BORDE_BAJO), 0, 1)
            s._f[i] = u * u * (3 - 2 * u)
            s._choque[i] = np.where(y < s.BORDE_ALTO + 0.08)[0]
        if s.ropa:
            s._ocultar_piel_bajo_ropa()

    # ---------------------------------------------------------------- 0. postura (bind)
    def _bind(s, clip):
        clave = (clip.reposo.round(5).tobytes(), tuple(clip.nombres), 'v2', tuple(sorted(s.IGUALAR.items())), tuple(sorted(s.NEUTRO.items())))
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
            if n in s.NEUTRO and n not in s.IGUALAR:
                A[n] = Ap @ _rodrigues(np.array(s.NEUTRO[n], float))
            elif n in s.IGUALAR:
                A[n] = _fraccion(A[n] @ Ap.T, s.IGUALAR[n]) @ Ap
        Al = np.stack([A[n] for n in s.nombres]); Pl = np.stack([pA[n] for n in s.nombres])
        off = np.zeros_like(Pl)
        for j, p in enumerate(s.padres):
            if p >= 0: off[j] = Al[p].T @ (Pl[j] - Pl[p])
        ibm = np.stack([np.linalg.inv(_M(Al[j], Pl[j])) for j in range(len(Al))])
        def pierna(P, ix): return sum(np.linalg.norm(P[ix[a]] - P[ix[c]]) for a, c in (('LeftLeg', 'LeftShin'), ('LeftShin', 'LeftFoot')))
        k = pierna(Pl, ix) / pierna(clip.reposo, clip.idx)
        b = dict(A=Al, P=Pl, off=off, ibm=ibm, k=k, hips0=clip.reposo[clip.idx['Hips']])
        s._cache[clave] = b; return b

    # ---------------------------------------------------------------- pesos del cuello: reparto suave Neck1/Neck2
    def _pesos(s):
        """MaxSoma reparte el peso Neck de mixamo entre Neck1 y Neck2 con un corte seco a la altura de Neck2; con el cuello
        recto ese corte hacía un escalón en la nuca. Aquí el reparto es gradual (6 cm de transición)."""
        super()._pesos()
        nodos = s.J['nodes']
        k_neck = [k for k, j in enumerate(s.base.joints) if nodos[j]['name'].endswith(':Neck') or nodos[j]['name'] == 'Neck'][0]
        y2 = s.pA['Neck2'][1]; n1, n2 = s.idx['Neck1'], s.idx['Neck2']
        for i, p in enumerate(s.prims):
            Jn, Wn = s.Jn[i], s.Wn[i]
            esq = p['J'][:, :min(4, Jn.shape[1])] == k_neck
            if not esq.any():
                continue
            u = np.clip((p['P'][:, 1] - (y2 - 0.03)) / 0.06, 0, 1); f = u * u * (3 - 2 * u)
            Jn = Jn.copy(); W = Wn.copy()
            w = np.where(esq, W[:, :esq.shape[1]], 0).sum(1)
            Jn[:, :esq.shape[1]] = np.where(esq, n1, Jn[:, :esq.shape[1]])
            W[:, :esq.shape[1]] = np.where(esq, W[:, :esq.shape[1]] * (1 - f)[:, None], W[:, :esq.shape[1]])
            s.Jn[i] = np.c_[Jn, np.full(len(Jn), n2)]; s.Wn[i] = np.c_[W, w * f]

    # ---------------------------------------------------------------- 1. piel tapada
    def _ocultar_piel_bajo_ropa(s):
        afuera = [i for i in s.ropa if not s.prims[i]['mname'].startswith('Poloche_Interior')]
        if not afuera:
            return
        p = s.prims[afuera[0]]
        P, N = p['P'], p['N'] / np.maximum(np.linalg.norm(p['N'], axis=1, keepdims=True), 1e-9)
        F = p['idx'].reshape(-1, 3)
        # aberturas = bordes de la cara de afuera (cuello, mangas, bajo)
        e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
        # vértices repetidos por las costuras de UV: se agrupan por posición
        clave = np.unique(np.round(P, 5), axis=0, return_inverse=True)[1].ravel()
        e = np.sort(clave[e], 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        borde_c = np.unique(u[c == 1])
        rep = np.zeros(clave.max() + 1, int); rep[clave] = np.arange(len(P))
        borde = P[rep[borde_c]]
        borde = borde[borde[:, 1] > 1.10]      # solo cuello y mangas (debajo del bajo está el pantalón, que tapa)
        s.piel_oculta = 0
        for q in s.prims:
            if not q['mname'].startswith(('Std_Skin_Body', 'Std_Skin_Arm')):
                continue
            V = q['P']
            j, d = _vecino(P, V)
            sd = np.einsum('ij,ij->i', V - P[j], N[j])
            _, db = _vecino(borde, V)
            tapada = (sd < -s.PIEL_ADENTRO) & (d < 0.09) & (db > s.PIEL_ABERTURA)
            tri = q['idx'].reshape(-1, 3)
            quita = tapada[tri].sum(1) >= 2
            if 'idx_sin_ropa' not in q:
                q['idx_sin_ropa'] = q['idx']
            q['idx'] = tri[~quita].ravel()
            s.piel_oculta += int(quita.sum())

    # ---------------------------------------------------------------- 2 y 3. física del borde + choque con el pantalón
    def pose(s, clip, t, bucle=False, apply=True):
        s._t = float(t); s._clip = id(clip)
        return super().pose(clip, t, bucle, apply)

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if s.MANOS_COMO_ANTES:
            s._manos_como_antes(clip, t, bucle, G, Pm, b)
        return G, Pm, b

    GLB_REFERENCIA = None     # glb del Max con que se hicieron los agarres (por defecto Personajes/Max/max_v1.glb)

    def _posiciones_maxsoma(s, clip, t, bucle):
        """Posiciones de las articulaciones tal como las pone MaxSoma con max_v1 (donde agarres_ik dejó las manos),
        pasadas al marco de este glb."""
        if getattr(s, '_ref', None) is None:
            import os
            from .rig import MAX_GLB
            ruta = s.GLB_REFERENCIA or MAX_GLB
            s._ref = MaxSoma(glb=ruta) if os.path.exists(ruta) else False
        if s._ref is False:
            b = MaxSoma._bind(s, clip); Gc, P = clip.globales(t, bucle)
            orden = [clip.idx[n] for n in s.nombres]; G = Gc[orden]; Ph = P[clip.idx['Hips']]
            Pm = np.zeros((len(s.nombres), 3)); ih = s.idx['Hips']
            for j, p in enumerate(s.padres):
                if j == ih: Pm[j] = b['P'][ih] + b['k'] * (Ph - b['hips0'])
                elif p < 0: Pm[j] = b['P'][j]
                else: Pm[j] = Pm[p] + G[p] @ b['off'][j]
            return Pm
        r = s._ref
        _, Pr, _ = r._sin_bloqueo(clip, t, bucle)
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        mundo = (Pr @ Cz.T - np.array([0, 0, r.base.zoff])) * r.base.scale
        return (mundo / s.base.scale + np.array([0, 0, s.base.zoff])) @ Cz       # Cz es ortogonal: inversa = transpuesta

    def _manos_como_antes(s, clip, t, bucle, G, Pm, b):
        P0 = s._posiciones_maxsoma(clip, t, bucle)
        ix = s.idx
        if not hasattr(s, '_desc'):
            s._desc = {}
            for L in ('Left', 'Right'):
                a = ix[L + 'Arm']; d = [a]
                for j in range(len(s.nombres)):
                    if j not in d and s.padres[j] in d: d.append(j)
                s._desc[L] = sorted(d)
        for L in ('Left', 'Right'):
            iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
            A_, E, H = Pm[iA], Pm[iF], Pm[iH]; T = P0[iH]
            l1, l2 = np.linalg.norm(E - A_), np.linalg.norm(H - E)
            dv = T - A_; d = min(np.linalg.norm(dv), (l1 + l2) * .9995); e = _norm_(dv)
            x = (l1 * l1 - l2 * l2 + d * d) / (2 * d); hh = np.sqrt(max(l1 * l1 - x * x, 0.0))
            ref = P0[iF] - P0[iA]                                  # codo hacia el mismo lado que antes
            pol = ref - e * (ref @ e)
            if np.linalg.norm(pol) < 1e-6: pol = (E - A_) - e * ((E - A_) @ e)
            pol = _norm_(pol)
            E2 = A_ + x * e + hh * pol; H2 = A_ + d * e
            R1 = _swing(E - A_, E2 - A_); G[iA] = R1 @ G[iA]
            Gf = R1 @ G[iF]; H1 = E2 + Gf @ b['off'][iH]; G[iF] = _swing(H1 - E2, H2 - E2) @ Gf
            for j in s._desc[L]:
                if j == iA: continue
                Pm[j] = Pm[s.padres[j]] + G[s.padres[j]] @ b['off'][j]

    def pose_mats(s, G, Pm, b, apply=True):
        if not s.ropa or not apply:
            return super().pose_mats(G, Pm, b, apply)
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        s.G, s.Wp = G, Pm
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        Vs, Ns = [], []
        for p, Jn, Wn in zip(s.prims, s.Jn, s.Wn):
            Mv = np.einsum('vk,vkij->vij', Wn, K[Jn])
            V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p.get('Pm', p['P'])) + Mv[:, :3, 3]
            Nn = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N'])
            V = V @ Cz.T; Nn = Nn @ Cz.T
            V = (V - np.array([0, 0, s.base.zoff])) * s.base.scale
            Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            Vs.append(V); Ns.append(Nn)
        s._fisica(Vs, Ns)
        return [(V[p['idx']], Nn[p['idx']]) for V, Nn, p in zip(Vs, Ns, s.prims)]

    def _fisica(s, Vs, Ns):
        t = s._t if s._t is not None else 0.0
        est = s._estado
        nuevo = est is None or est['clip'] != s._clip or not (0 < t - est['t'] <= 0.25) or not s.FISICA
        if nuevo:
            est = s._estado = {'clip': s._clip, 't': t, 'x': {i: Vs[i].copy() for i in s.ropa},
                               'v': {i: np.zeros_like(Vs[i]) for i in s.ropa}, 'obj': {i: Vs[i].copy() for i in s.ropa}}
        else:
            dt = t - est['t']
            w = 2 * np.pi * s.FRECUENCIA; c = 2 * s.AMORTIGUA * w
            n = max(1, int(np.ceil(dt * 240)))
            h = dt / n
            for i in s.ropa:
                f = s._f[i]; m = f > 0
                x, v = est['x'][i][m], est['v'][i][m]
                o0, o1 = est['obj'][i][m], Vs[i][m]
                for k in range(n):
                    o = o0 + (o1 - o0) * ((k + 1) / n)
                    v = v + (w * w * (o - x) - c * v) * h
                    x = x + v * h
                dx = x - o1
                # la tela es una pieza: el desvío se alisa entre vecinos (sin esto el borde se ve deshilachado)
                if s._ady.get(i) is not None:
                    g = s._grupo[i]; ng = g.max() + 1; A, deg = s._ady[i]
                    full = np.zeros((len(f), 3)); full[m] = dx
                    dg = np.zeros((ng, 3)); np.add.at(dg, g, full); cg = np.bincount(g, minlength=ng)[:, None]; dg /= np.maximum(cg, 1)
                    for _ in range(3):
                        dg = 0.4 * dg + 0.6 * (A @ dg) / deg[:, None]
                    dx = dg[g][m]
                    vfull = np.zeros((len(f), 3)); vfull[m] = v
                    vg = np.zeros((ng, 3)); np.add.at(vg, g, vfull); vg /= np.maximum(cg, 1)
                    for _ in range(2):
                        vg = 0.5 * vg + 0.5 * (A @ vg) / deg[:, None]
                    v = vg[g][m]
                lim = np.linalg.norm(dx, axis=1, keepdims=True)
                dx = dx * np.minimum(1, s.DESVIO_MAX / np.maximum(lim, 1e-9))
                x = o1 + dx
                est['x'][i][m] = x; est['v'][i][m] = v
                est['obj'][i] = Vs[i].copy()
                Vs[i][m] = o1 + f[m, None] * dx
            est['t'] = t
        # choque con las piernas (cápsulas cadera->rodilla del grosor del pantalón): la tela nunca queda dentro del muslo
        s._choque_piernas(Vs, est, Ns)

    RADIO_MUSLO = 0.125      # m: radio del muslo con el pantalón puesto (medido en el pantalón ancho de Max)

    def _capsulas_bind(s):
        if getattr(s, '_c0', None) is not None:
            return
        s._c0 = {}
        for i in s.ropa:
            P = s.prims[i]['P']; cs = []
            for L in ('Left', 'Right'):
                a, b = s.pA[L + 'Leg'], s.pA[L + 'Shin']
                cs.append(s._dist_seg(P, a, b) - s.RADIO_MUSLO)
            s._c0[i] = np.stack(cs, 1) * s.base.scale

    @staticmethod
    def _dist_seg(P, a, b):
        d = b - a; t = np.clip(((P - a) @ d) / (d @ d), 0, 1)
        return np.linalg.norm(P - (a + t[:, None] * d), axis=1)

    def _choque_piernas(s, Vs, est, Ns=None):
        s._capsulas_bind()
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        def mundo(q): return (Cz @ q - np.array([0, 0, s.base.zoff])) * s.base.scale
        # la cara de adentro se mueve igual que la de afuera (par más cercano en el enlace): la tela no se rasga
        if not hasattr(s, '_par'):
            s._par = {}
            afuera = [i for i in s.ropa if not s.prims[i]['mname'].startswith('Poloche_Interior')]
            if afuera and cKDTree is not None:
                ia = afuera[0]
                for i in s.ropa:
                    if i != ia:
                        s._par[i] = (ia, cKDTree(s.prims[ia]['P']).query(s.prims[i]['P'])[1])
        e_hechos = {}
        orden = sorted(s.ropa, key=lambda i: i in s._par)
        for i in orden:
            if i in s._par and s._par[i][0] in e_hechos:
                ia, par = s._par[i]; e = e_hechos[ia][par]
                if np.any(e):
                    Vs[i] += e
                    if est is not None: est['x'][i] += e
                continue
            sel = s._choque[i]
            margen = (s.MARGEN_ADENTRO if s.prims[i]['mname'].startswith('Poloche_Interior') else s.MARGEN_AFUERA)
            e = np.zeros_like(Vs[i])
            for k, L in enumerate(('Left', 'Right')):
                a, b = mundo(s.Wp[s.idx[L + 'Leg']]), mundo(s.Wp[s.idx[L + 'Shin']])
                q = Vs[i][sel] + e[sel]
                d = b - a; t = np.clip(((q - a) @ d) / (d @ d), 0, 1); c = a + t[:, None] * d
                r = q - c; dist = np.linalg.norm(r, axis=1); R = s.RADIO_MUSLO * s.base.scale
                quiero = np.minimum(s._c0[i][sel, k], margen) + R
                mal = (dist < quiero) & (dist > 1e-6)
                if mal.any():
                    e[sel[mal]] += r[mal] / dist[mal, None] * (quiero[mal] - dist[mal])[:, None]
            # además, el pantalón real (en la pose): la tela queda por fuera de la cintura y la cadera del pantalón
            if Ns is not None and s.pantalon and cKDTree is not None:
                Pp = np.concatenate([Vs[j] for j in s.pantalon]); Np_ = np.concatenate([Ns[j] for j in s.pantalon])
                if not hasattr(s, '_pant_alto'):
                    yb = np.concatenate([s.prims[j]['P'][:, 1] for j in s.pantalon]); s._pant_alto = yb > 0.80
                Pp, Np_ = Pp[s._pant_alto], Np_[s._pant_alto]
                q = Vs[i][sel] + e[sel]
                d, j = cKDTree(Pp).query(q, distance_upper_bound=0.06)
                ok = np.isfinite(d)
                if ok.any():
                    jj = j[ok]; si = sel[ok]
                    sd = np.einsum('ij,ij->i', q[ok] - Pp[jj], Np_[jj])
                    mg = s.MARGEN_ADENTRO if s.prims[i]['mname'].startswith('Poloche_Interior') else s.MARGEN_AFUERA
                    mal = sd < mg
                    e[si[mal]] += Np_[jj[mal]] * (mg - sd[mal])[:, None]
            if not np.any(e):
                e_hechos[i] = e
                continue
            g = s._grupo[i]; ng = g.max() + 1
            eg = np.zeros((ng, 3)); np.add.at(eg, g, e); cg = np.bincount(g, minlength=ng)[:, None]; eg /= np.maximum(cg, 1)
            if s._ady.get(i) is not None:
                A, deg = s._ady[i]
                for _ in range(4):
                    sm = (A @ eg) / deg[:, None]
                    eg = np.where((np.linalg.norm(sm, axis=1) > np.linalg.norm(eg, axis=1))[:, None], sm, eg)
            e = eg[g]
            e_hechos[i] = e
            Vs[i] += e
            if est is not None:
                est['x'][i] += e
                tocado = np.linalg.norm(e, axis=1) > 0
                nrm = e[tocado] / np.linalg.norm(e[tocado], axis=1, keepdims=True)
                vv = est['v'][i][tocado]; vn = np.einsum('ij,ij->i', vv, nrm)
                est['v'][i][tocado] = vv - nrm * np.minimum(vn, 0)[:, None]
