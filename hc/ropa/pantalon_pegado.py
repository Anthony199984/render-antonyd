"""Pantalón pegado: al agacharse hondo (sentadilla, cuclillas, levantar algo pesado del piso) el pantalón ancho de Max se
separaba de la piel hasta 13-20 cm y se inflaba como un globo (de pie: 5-8 cm). Esta capa, cuadro por cuadro, mide la
distancia de cada punto del pantalón a la piel que tiene debajo y:
  - si se separó más de lo que estaba de pie (+ HOLGURA), lo acerca a la piel;
  - si la piel lo atraviesa, lo empuja afuera (MIN_TELA).
De pie y caminando no cambia nada (el pantalón ya está dentro de su holgura).

Sirve para cualquier versión de Max del motor, sin tocar archivos viejos:
    from pantalon_pegado import con_pantalon_pegado
    from soma.max_v17 import MaxCaraV17
    Max = con_pantalon_pegado(MaxCaraV17)          # la misma clase, con el pantalón pegado
    m = Max(glb=...)
    Max.PANTALON_PEGADO = False                    # para apagarlo
"""
import numpy as np


class _PantalonPegado:
    PANTALON_PEGADO = True
    HOLGURA = 0.015       # m que el pantalón puede separarse de más respecto a como está de pie
    MIN_TELA = 0.003      # m: la piel nunca asoma; la tela queda al menos así de afuera
    CAMISETA_SIGUE = False     # (ya no hace falta: la camiseta se arregla en camiseta_planchada.py)
    CAMISETA_RADIO = 0.12  # m: la camiseta a esta distancia del pantalón lo acompaña
    SUAVIZAR = 3          # pasadas de suavizado del empuje (que no queden arrugas en escalón)

    PIERNAS_SEPARADAS = True   # el interior de cada pierna del pantalón va SOLO con su pierna (no hace tela entre rodillas)
    CENTRO = 0.09              # m desde la costura del medio: más allá, cada lado es de su pierna

    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        if s.PANTALON_PEGADO:
            if s.PIERNAS_SEPARADAS:
                s._pp_pesos()
            s._pp_preparar()

    def _pp_pesos(s):
        """Antes 1 de cada 5 puntos del pantalón iba pesado a las DOS piernas a la vez: al abrir las rodillas en una
        sentadilla, esos puntos quedaban a mitad de camino y formaban una tela de rodilla a rodilla donde se metían los
        brazos. Ahora: lejos del centro, el peso de la otra pierna pasa a la pierna propia; en la costura del medio,
        el peso de las piernas pasa a la cadera (la entrepierna sigue a la pelvis)."""
        ix = s.idx; lis = lambda x: (lambda y: y * y * (3 - 2 * y))(np.clip(x, 0, 1))
        lado = {}
        for L, O in (('Left', 'Right'), ('Right', 'Left')):
            for b in ('Leg', 'Shin', 'Foot', 'ToeBase'):
                if L + b in ix and O + b in ix: lado[ix[L + b]] = (L, ix[O + b])
        s._pp_sep = {}
        cx = (s.pA['LeftLeg'][0] + s.pA['RightLeg'][0]) / 2
        sl = np.sign(s.pA['LeftLeg'][0] - cx); hips = ix['Hips']
        for i, p in enumerate(s.prims):
            if not p['mname'].startswith('Pants'): continue
            J, W = s.Jn[i].copy(), s.Wn[i].copy()
            esL = np.vectorize(lambda j: j in lado and lado[j][0] == 'Left')(J)
            esR = np.vectorize(lambda j: j in lado and lado[j][0] == 'Right')(J)
            wl, wr = (W * esL).sum(1), (W * esR).sum(1)
            mezcla = (wl > 1e-3) & (wr > 1e-3)
            if not mezcla.any(): continue
            dx = (p['P'][:, 0] - cx) * sl * s.base.scale          # + del lado izquierdo de Max
            a = lis((np.abs(dx) - 0.01) / (s.CENTRO - 0.01))      # 0 en la costura, 1 ya en la pierna
            propio_izq = dx > 0
            for v in np.where(mezcla)[0]:
                for k in range(J.shape[1]):
                    j = J[v, k]
                    if j not in lado or W[v, k] == 0: continue
                    otro = (lado[j][0] == 'Left') != propio_izq[v]
                    if otro:
                        mover = W[v, k] * a[v]; W[v, k] -= mover
                        jj = lado[j][1]; kk = np.where(J[v] == jj)[0]
                        if len(kk): W[v, kk[0]] += mover
                        else: J[v, k] = jj; W[v, k] += mover   # el hueso propio no estaba: este hueco pasa a ser él
                # costura del medio: lo de las piernas pasa a la cadera
                c = 1 - a[v]
                if c > 0:
                    pierna = np.array([j in lado for j in J[v]])
                    mover = (W[v] * pierna).sum() * c * 0.7
                    W[v][pierna] *= (1 - c * 0.7)
                    kh = np.where(J[v] == hips)[0]
                    if len(kh): W[v, kh[0]] += mover
                    else:
                        k0 = np.argmin(W[v]); J[v, k0] = hips; W[v, k0] = mover
            W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
            s._pp_sep[i] = (J, W)

    def _pp_preparar(s):
        from scipy.spatial import cKDTree
        s._pp_piel = [i for i, p in enumerate(s.prims) if p['mname'].startswith(('Std_Skin_Body', 'Std_Skin_Leg'))]
        s._pp_pant = [i for i, p in enumerate(s.prims) if p['mname'].startswith('Pants')]
        if not s._pp_piel or not s._pp_pant:
            s._pp_pant = []; return
        X = np.concatenate([s.prims[i].get('Pm', s.prims[i]['P']) for i in s._pp_piel])
        N = np.concatenate([s.prims[i]['N'] for i in s._pp_piel])
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        arbol = cKDTree(X)
        s._pp_g0 = {}; s._pp_vec = {}
        for i in s._pp_pant:
            P = s.prims[i].get('Pm', s.prims[i]['P'])
            d, ix = arbol.query(P)
            s._pp_g0[i] = d * s.base.scale                          # separación de pie
            # vecinos de la malla (para suavizar el empuje), uniendo puntos repetidos en las costuras
            T = s.prims[i]['idx'].reshape(-1, 3)
            _, uni = np.unique(np.round(P, 6), axis=0, return_inverse=True); uni = uni.ravel()
            Tu = uni[T]; nU = uni.max() + 1
            A = np.concatenate([Tu[:, [0, 1]], Tu[:, [1, 2]], Tu[:, [2, 0]]]); A = np.concatenate([A, A[:, ::-1]])
            s._pp_vec[i] = (uni, A, nU)

    DESDE, HASTA = 35.0, 70.0   # grados de muslo levantado: aquí empieza / del todo (parado y caminando: nada)

    def _pp_cuanto(s, Pm):
        ix = s.idx; abajo = Pm[ix['Hips']] - Pm[ix['Spine2']]; abajo /= np.linalg.norm(abajo) + 1e-12; a = 0.0
        for L in ('Left', 'Right'):
            m_ = Pm[ix[L + 'Shin']] - Pm[ix[L + 'Leg']]; m_ /= np.linalg.norm(m_) + 1e-12
            a = max(a, np.degrees(np.arccos(np.clip(m_ @ abajo, -1, 1))))
        x = np.clip((a - s.DESDE) / (s.HASTA - s.DESDE), 0, 1); return float(x * x * (3 - 2 * x))

    def _pp_todos(s, G, Pm, b, lista, pesos=None):
        from soma.rig import _M
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]]); out = {}
        for i in lista:
            p = s.prims[i]; J, W = (pesos or {}).get(i, (s.Jn[i], s.Wn[i]))
            Mv = np.einsum('vk,vkij->vij', W, K[J])
            V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p.get('Pm', p['P'])) + Mv[:, :3, 3]
            Nn = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N'])
            V = (V @ Cz.T - [0, 0, s.base.zoff]) * s.base.scale; Nn = Nn @ Cz.T
            out[i] = (V, Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9))
        return out

    def pp_empuje(s, G, Pm, b):
        """{prim del pantalón: desplazamiento por punto (m, espacio final)}."""
        from scipy.spatial import cKDTree
        c = s._pp_cuanto(Pm)
        if c < 1e-3: return {}
        R = s._pp_todos(G, Pm, b, s._pp_piel + s._pp_pant)
        sep = s._pp_todos(G, Pm, b, [i for i in s._pp_pant if i in getattr(s, '_pp_sep', {})], s._pp_sep) if getattr(s, '_pp_sep', None) else {}
        X = np.concatenate([R[i][0] for i in s._pp_piel]); N = np.concatenate([R[i][1] for i in s._pp_piel])
        arbol = cKDTree(X); out = {}
        for i in s._pp_pant:
            V0 = R[i][0]; V = V0 + c * (sep[i][0] - V0) if i in sep else V0
            d, ix = arbol.query(V)
            g = np.einsum('vi,vi->v', V - X[ix], N[ix])
            lim = s._pp_g0[i] + s.HOLGURA
            # 1) muy separado: se acerca en línea recta a la piel más cercana
            u = (V - X[ix]) / np.maximum(d, 1e-9)[:, None]
            D = np.where((d > lim)[:, None], u * (lim - d)[:, None], 0.0)
            # 2) la piel lo atraviesa: afuera por la normal de la piel
            D += np.where((g < s.MIN_TELA)[:, None], (s.MIN_TELA - g)[:, None] * N[ix], 0.0)
            uni, A, nU = s._pp_vec[i]
            Du = np.zeros((nU, 3)); cnt = np.zeros(nU)
            np.add.at(Du, uni, D); np.add.at(cnt, uni, 1); Du /= np.maximum(cnt, 1)[:, None]
            deg = np.bincount(A[:, 0], minlength=nU).astype(float)
            for _ in range(s.SUAVIZAR):
                vec = np.zeros_like(Du); np.add.at(vec, A[:, 0], Du[A[:, 1]])
                Du = 0.5 * Du + 0.5 * vec / np.maximum(deg, 1)[:, None]
            out[i] = (V - V0) + c * Du[uni]
        s._pp_V0 = {i: R[i][0] for i in s._pp_pant}
        return out

    def pose_mats(s, G, Pm, b, apply=True):
        res = super().pose_mats(G, Pm, b, apply)
        if not (s.PANTALON_PEGADO and apply and getattr(s, '_pp_pant', None)):
            return res
        emp = s.pp_empuje(G, Pm, b)
        for i, D in emp.items():
            V, Nn = res[i]
            if len(V):
                idx = s.prims[i]['idx']; V2 = V + D[idx]
                if hasattr(s, '_normales') and hasattr(s, '_malla_tela'):      # luces bien puestas sobre la forma nueva
                    n = len(s.prims[i]['P']); Vv = np.zeros((n, 3)); Vv[idx] = V2; Nv = np.zeros((n, 3)); Nv[idx] = Nn
                    N2 = s._normales(i, Vv, Nv)[idx]
                    w = np.clip(np.linalg.norm(D[idx], axis=1) / 0.01, 0, 1)[:, None]
                    Nn = Nn * (1 - w) + N2 * w; Nn = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
                res[i] = (V2, Nn)
        if emp and s.CAMISETA_SIGUE:
            # la camiseta que está apoyada sobre el pantalón (el borde de abajo, en el regazo) se mueve con él;
            # si no, quedaba flotando donde antes estaba la tela entre las rodillas (pedazos blancos en la entrepierna)
            from scipy.spatial import cKDTree
            P0 = np.concatenate([s._pp_V0[i] for i in emp]); D0 = np.concatenate([emp[i] for i in emp])
            arbol = cKDTree(P0)
            for k, p in enumerate(s.prims):
                if not p['mname'].startswith('Poloche') or not len(res[k][0]): continue
                V, Nn = res[k]; d, ix = arbol.query(V, distance_upper_bound=s.CAMISETA_RADIO)
                ok = np.isfinite(d)
                if ok.any():
                    f = np.zeros(len(V)); f[ok] = np.clip(1 - d[ok] / s.CAMISETA_RADIO, 0, 1) ** 0.5
                    V = V.copy(); V[ok] += f[ok, None] * D0[ix[ok]]
                    res[k] = (V, Nn)
        return res


def con_pantalon_pegado(Clase):
    """Devuelve la misma clase de Max con el pantalón pegado encima (no modifica la clase original)."""
    return type(Clase.__name__ + 'PP', (_PantalonPegado, Clase), {})
