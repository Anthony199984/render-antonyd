"""Max V17 = Max V16 (límites del cuerpo) + CONTACTO CON EL PISO + ROPA AL SENTARSE. Archivo NUEVO: no cambia max_v16.py.

Por qué (prueba del 2 nov 2026 con 44 animaciones nuevas de Kimodo):
 - al tropezar y caer boca abajo, las manos se quedaban en el aire, a ~10 cm del piso, en vez de apoyarse;
 - sentado en el piso flotaba (una mano metida en el piso subía todo el cuerpo ~15 cm) y los pies se metían en el piso;
 - al sentarse o ponerse en cuclillas, el borde de abajo de la camiseta colgaba hasta el piso y se doblaba al revés
   debajo de los muslos (los "pedazos blancos"), y unos pedacitos de piel de los glúteos asomaban bajo el pantalón.

 1. NADA DENTRO DEL PISO (PISO_SOLIDO): si el tronco, la cabeza o las piernas quedan debajo del piso, todo el cuerpo
    sube lo justo para apoyarse encima (los brazos no levantan el cuerpo: las manos se acomodan solas). Sentado o
    acostado, un pie que quedó dentro del piso sube doblando un poco la pierna. Parado no cambia nada.
 2. MANOS QUE SE APOYAN (MANOS_AL_PISO): cuando Max está en el piso (caído, gateando, hincado, acostado) y una mano
    quieta queda cerca del piso (a menos de ~18 cm), el brazo la baja hasta apoyarla (y si se hundía, la sube). La mano
    conserva su giro; una mano que se mueve (gateando, en el aire) no se toca. Nunca en la mano que agarra o sostiene.
 3. SENTADO DE VERDAD (SENTADO_APOYA): sentado en el piso, si las nalgas quedaron en el aire (hasta 15 cm), Max baja
    hasta apoyarlas, sin meter los pies en el piso.
 4. CAMISETA AL SENTARSE (CAMISETA_SENTADO): con los muslos levantados (sentado, en cuclillas) la parte de abajo de la
    camiseta se recoge hacia la cintura como la tela de verdad; si el muslo se mete en la camiseta, la tela queda por
    encima del pantalón; si toca el piso, queda apoyada encima. Parado o caminando no cambia (menos de 1 cm).
 5. PANTALÓN (PANTALON_PIEL): la cintura y los costados de arriba del pantalón siguen a la piel de Max al doblarse; la
    entrepierna y el interior del muslo siguen con las piernas. Se esconden los pedacitos de piel bajo el pantalón.

    from soma.max_v17 import MaxCaraV17, PistaCaraV17
    m = MaxCaraV17(glb='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV17.desde_escena('<carpeta de salida de escena5.py o escena6.py>', '<escena.json>')
Se apagan con MaxCaraV17.PISO_SOLIDO / MANOS_AL_PISO / SENTADO_APOYA / CAMISETA_SENTADO / PANTALON_PIEL = False.
Ajustes: PISO_Z (altura del piso de la escena, m; 0 por defecto), MANO_CERCA, CUERPO_BAJO, SENTADO_CADERA,
SENTADO_HUECO, FALDA_SUBE, MUSLO_HOLGURA.
"""
import numpy as np

from .max_v16 import MaxCaraV16, MaxV16, PistaCaraV16


class PistaCaraV17(PistaCaraV16):
    """Igual que PistaCaraV16."""


def _suave(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def _norm(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def _swing(a, b):
    a = _norm(a); b = _norm(b); v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        p = _norm(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < .9 else np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class _Piso:
    PISO_SOLIDO = True
    MANOS_AL_PISO = True
    SENTADO_APOYA = True
    SENTADO_CADERA = 0.55   # m: cadera más baja que esto (y nalgas casi en el piso) = sentado en el piso
    SENTADO_HUECO = 0.15    # m: hasta este hueco entre nalgas y piso se baja a apoyarlas
    PISO_Z = 0.0            # altura del piso (m, como joint_pos)
    MANO_CERCA = 0.18       # una mano a menos de esto del piso (con el cuerpo en el piso) se apoya
    CUERPO_BAJO = 0.80      # cadera más baja que esto = Max está en el piso (del todo a 0,65 m)
    MANO_QUIETA = 0.35      # m/s: una mano que se mueve más rápido que esto de lado no se apoya (va por el aire)
    PIEL_CADA = 6           # se mira 1 de cada N puntos de la piel (más chico = más exacto y más lento)

    # puntos de la piel pegados a su hueso (en el espacio del hueso), para medir alturas rápido
    def _muestras(s, b):
        c = s.__dict__.setdefault('_muestras_piel', {})
        if id(b) in c:
            return c[id(b)]
        J = []; X = []; T = []; Q = []; Z = []
        for p, Jn, Wn in zip(s.prims, s.Jn, s.Wn):
            if 'Cornea' in p.get('mname', '') or 'Eye' in p.get('mname', ''):
                continue
            P = p.get('Pm', p['P'])
            vis = np.unique(p['idx'])                          # solo la piel que se dibuja (la tapada por la ropa no)
            if len(vis) == 0:
                continue
            sel = vis[::s.PIEL_CADA]
            jd = Jn[sel, np.argmax(Wn[sel], axis=1)]
            J.append(jd); X.append(np.c_[P[sel], np.ones(len(sel))]); T.append(np.full(len(sel), p['mname'].startswith('Poloche')))
            Q.append(np.full(len(sel), p['mname'] == 'Pants')); Z.append(np.full(len(sel), p['mname'].startswith(('Sneaker', 'Media', 'Std_Nails'))))
        J = np.concatenate(J); X = np.concatenate(X); tela = np.concatenate(T); pant = np.concatenate(Q); zap = np.concatenate(Z)
        loc = np.einsum('vij,vj->vi', b['ibm'][J], X)[:, :3]          # en el espacio de su hueso
        mano = {L: np.isin(J, [j for j, n in enumerate(s.nombres) if n.startswith(L + 'Hand')]) for L in ('Left', 'Right')}
        # los zapatos no cuentan para "dentro del piso": de los pies se encarga el bloqueo de pies (parado no cambia nada)
        for L in ('Left', 'Right'):     # de qué pierna es cada punto (zapato, media, uña del pie: por su hueso)
            mano['pie' + L] = np.isin(J, [j for j, n in enumerate(s.nombres) if n.startswith(L) and ('Foot' in n or 'Toe' in n or 'Shin' in n)])
        mano['nalgas'] = pant & np.isin(J, [s.idx[n] for n in ('Hips', 'LeftLeg', 'RightLeg')])
        mano['pies'] = zap | np.isin(J, [j for j, n in enumerate(s.nombres) if 'Foot' in n or 'Toe' in n])
        mano['cuerpo'] = ~tela & ~zap & ~np.isin(J, [j for j, n in enumerate(s.nombres)
                                                     if 'Foot' in n or 'Toe' in n or 'Arm' in n or 'Hand' in n])
        c[id(b)] = (J, loc, mano)
        return c[id(b)]

    def _alturas(s, G, Pm, b):
        J, loc, mano = s._muestras(b)
        x = np.einsum('vij,vj->vi', G[J], loc) + Pm[J]                 # espacio glTF (Y arriba)
        return (x[:, 1] - s.base.zoff) * s.base.scale - s.PISO_Z, mano

    def _altura_j(s, Pm, j):
        return (Pm[j, 1] - s.base.zoff) * s.base.scale - s.PISO_Z

    def _hijos_de(s, j):
        c = s.__dict__.setdefault('_desc', {})
        if j not in c:
            d = {j}
            for k, p in enumerate(s.padres):
                if p in d:
                    d.add(k)
            d.discard(j); c[j] = sorted(d)
        return c[j]

    def _ik_brazo(s, G, Pm, b, L, objetivo):
        ix = s.idx; iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
        A, E, H = Pm[iA], Pm[iF], Pm[iH]; l1, l2 = np.linalg.norm(E - A), np.linalg.norm(H - E)
        d_v = objetivo - A; d = min(np.linalg.norm(d_v), (l1 + l2) * .999); e = _norm(d_v)
        x = (l1 * l1 - l2 * l2 + d * d) / (2 * d); hh = np.sqrt(max(l1 * l1 - x * x, 0.0))
        pol = (E - A) - e * np.dot(E - A, e)
        pol = _norm(pol) if np.linalg.norm(pol) > 1e-6 else _norm(np.cross(e, [0, 1.0, 0]))
        E2 = A + x * e + hh * pol; H2 = A + d * e
        Gh = G[iH].copy()                                               # la mano conserva su giro
        R1 = _swing(E - A, E2 - A); G[iA] = R1 @ G[iA]
        Gs = R1 @ G[iF]; H1 = E2 + Gs @ b['off'][iH]; G[iF] = _swing(H1 - E2, H2 - E2) @ Gs
        Pm[iF] = Pm[iA] + G[iA] @ b['off'][iF]; Pm[iH] = Pm[iF] + G[iF] @ b['off'][iH]
        G[iH] = Gh                                                      # los dedos (globales) tampoco giran
        for k in s._hijos_de(iH):
            Pm[k] = Pm[s.padres[k]] + G[s.padres[k]] @ b['off'][k]

    def _quieta(s, clip, L):
        c = s.__dict__.setdefault('_manos_quietas', {})
        k = (id(clip), L)
        if k not in c:
            from scipy.ndimage import gaussian_filter1d
            p = clip.P[:, clip.idx[L + 'Hand']]
            v = np.r_[0, np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * clip.fps]
            v = gaussian_filter1d(v, 1.5, mode='nearest')
            c[k] = gaussian_filter1d(_suave((s.MANO_QUIETA - v) / 0.15), 1.5, mode='nearest')
        return c[k]

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if not (s.PISO_SOLIDO or s.MANOS_AL_PISO):
            return G, Pm, b
        G = G.copy(); Pm = Pm.copy()
        h, mano = s._alturas(G, Pm, b)
        # 1. nada dentro del piso
        hc = h[mano['cuerpo']].min()
        if s.PISO_SOLIDO and hc < 0:
            Pm[:, 1] += -hc / s.base.scale
            h = h - hc
        # 1b. sentado en el piso: si las nalgas quedaron un poco en el aire, baja hasta apoyarlas (sin meter los pies)
        if s.SENTADO_APOYA:
            cadera = s._altura_j(Pm, s.idx['Hips'])
            nal = h[mano['nalgas']].min(); pie = h[mano['pies']].min()
            w = _suave((s.SENTADO_CADERA - cadera) / 0.10) * _suave((s.SENTADO_HUECO - nal) / 0.03)
            baja = w * max(0.0, min(nal, pie))
            if baja > 1e-4:
                Pm[:, 1] -= baja / s.base.scale; h = h - baja
        # 1c. sentado o acostado: un pie que quedó dentro del piso sube (la pierna se dobla un poco), sin mover el cuerpo
        if s.PISO_SOLIDO:
            cadera = s._altura_j(Pm, s.idx['Hips'])
            wb = _suave((s.SENTADO_CADERA - cadera) / 0.10)
            if wb > 0:
                for L in ('Left', 'Right'):
                    sel = mano['pies'] & mano['pie' + L]
                    hp = h[sel].min() if sel.any() else 0.0
                    if hp < -0.003:
                        obj = Pm[s.idx[L + 'Foot']].copy(); obj[1] += wb * (-hp) / s.base.scale
                        s._ik_pierna(G, Pm, b, L, obj)
                        h = s._alturas(G, Pm, b)[0]
        # 2. manos que se apoyan
        if s.MANOS_AL_PISO:
            cadera = s._altura_j(Pm, s.idx['Hips'])
            bajo = _suave((s.CUERPO_BAJO - cadera) / 0.15)
            if bajo > 0:
                f = int(np.clip(round(t * clip.fps), 0, clip.n - 1))
                for L in ('Left', 'Right'):
                    hm = h[mano[L]].min()
                    w = bajo * _suave((s.MANO_CERCA - hm) / 0.06) * s._quieta(clip, L)[f]
                    if hasattr(s, '_peso_agarre'):
                        w *= 1.0 - s._peso_agarre(L, t)
                    if w < 1e-3 or abs(hm) < 0.003:
                        continue
                    for _ in range(2):                              # dos pasadas: la segunda corrige lo que faltó
                        obj = Pm[s.idx[L + 'Hand']].copy(); obj[1] -= w * hm / s.base.scale
                        s._ik_brazo(G, Pm, b, L, obj)
                        hm = s._alturas(G, Pm, b)[0][mano[L]].min()
                        if abs(hm) < 0.005:
                            break
        return G, Pm, b


class _PantalonComoLaPiel:
    """El pantalón sigue a la piel que tiene debajo. Antes iba pesado SOLO a los muslos y las espinillas (nada a la
    cadera): al sentarse o agacharse (muslo doblado 90-120°) los glúteos y la cintura del pantalón giraban con el muslo
    y el pantalón se inflaba como un globo. Ahora cada punto del pantalón copia los pesos de la piel más cercana
    (cuerpo y piernas), así se dobla igual que el cuerpo de Max. Parado y caminando casi no cambia."""
    PANTALON_PIEL = True
    PANTALON_VECINOS = 8
    PANT_DESDE, PANT_ANCHO = 0.12, 0.08     # m bajo la cadera: desde dónde el pantalón empieza a seguir a la piel
    PANT_CENTRO = 0.03                      # m: franja del centro de atrás que sigue con las piernas
    PANT_RODILLA = 0.0                      # m alrededor de la rodilla que siguen a la piel (0 = no; probado: no cambia)

    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        if s.PANTALON_PIEL:
            s._pantalon_como_piel()
        s._piel_bajo_pantalon()

    def _piel_bajo_pantalon(s):
        """Unos pocos triángulos de piel de los glúteos quedaban visibles debajo del pantalón (sentado asomaban)."""
        Ly = (s.pA['LeftLeg'][1] + s.pA['RightLeg'][1]) / 2
        for p in s.prims:
            if p['mname'].startswith('Std_Skin_Body'):
                T = p['idx'].reshape(-1, 3)
                fuera = (p['P'][T][:, :, 1] < Ly - 0.03 / s.base.scale).all(1)
                if fuera.any():
                    p['idx'] = T[~fuera].ravel()

    def _pantalon_como_piel(s):
        from scipy.spatial import cKDTree
        nJ = len(s.nombres)
        piel = [i for i, p in enumerate(s.prims) if p['mname'].startswith(('Std_Skin_Body', 'Std_Skin_Leg'))]
        pant = [i for i, p in enumerate(s.prims) if p['mname'] == 'Pants']
        if not piel or not pant:
            return
        X = np.concatenate([s.prims[i]['P'] for i in piel])
        D = np.concatenate([_denso(s.Jn[i], s.Wn[i], nJ) for i in piel])
        arbol = cKDTree(X)
        def lis(x):
            x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
        Ly = (s.pA['LeftLeg'][1] + s.pA['RightLeg'][1]) / 2; cx = (s.pA['LeftLeg'][0] + s.pA['RightLeg'][0]) / 2
        zc = s.pA['Hips'][2]; u = 1.0 / s.base.scale
        for i in pant:
            P = s.prims[i]['P']
            d, ix = arbol.query(P, k=s.PANTALON_VECINOS)
            w = 1.0 / np.maximum(d, 1e-4) ** 2; w /= w.sum(1, keepdims=True)
            Wd = np.einsum('vk,vkj->vj', w, D[ix])
            # solo la cintura, los glúteos y lo de arriba de los costados siguen a la piel; la entrepierna y el interior
            # del muslo siguen con las piernas (si no, al abrir las piernas el pantalón se estira como una tela entre ellas)
            alto = lis((P[:, 1] - (Ly - s.PANT_DESDE * u)) / (s.PANT_ANCHO * u))
            afuera = lis((np.abs(P[:, 0] - cx) - 0.03 * u) / (0.05 * u))
            atras = lis((zc - P[:, 2] - 0.0 * u) / (0.04 * u)) * lis((np.abs(P[:, 0] - cx) - s.PANT_CENTRO * u) / (0.03 * u) + 1)
            f = alto * np.maximum(afuera, atras)
            # la rodilla también como la piel (al doblarla mucho el pantalón hacía una bola)
            for L in ('Left', 'Right'):
                dk = np.linalg.norm(P - s.pA[L + 'Shin'], axis=1) * s.base.scale
                f = np.maximum(f, lis((s.PANT_RODILLA - dk) / 0.04))
            Wd = Wd * f[:, None] + _denso(s.Jn[i], s.Wn[i], nJ) * (1 - f[:, None])
            top = np.argsort(-Wd, axis=1)[:, :4]
            Wt = np.take_along_axis(Wd, top, 1); Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
            s.Jn[i] = top; s.Wn[i] = Wt


def _denso(Jn, Wn, nJ):
    D = np.zeros((len(Jn), nJ))
    np.add.at(D, (np.repeat(np.arange(len(Jn)), Jn.shape[1]), Jn.ravel()), Wn.ravel())
    return D


class _CamisetaSentado:
    """La camiseta al sentarse o agacharse: el borde de abajo va pesado a la cadera y, con el muslo levantado (sentado
    con las rodillas arriba, en cuclillas), el muslo la atravesaba y salía en pedazos blancos entre las piernas; además
    se metía en el piso. Ahora la tela que queda dentro del muslo se empuja afuera, por encima del pantalón, y la que
    queda debajo del piso se apoya en el piso. Parado o caminando no cambia nada (la tela nunca está dentro del muslo)."""
    CAMISETA_SENTADO = True
    MUSLO_HOLGURA = 0.012       # m de tela por encima del pantalón
    TELA_PISO = 0.004           # m: la tela que toca el piso queda apenas encima
    TELA_BUSCA = 0.06           # m: se mira el pantalón a esta distancia de cada punto de la camiseta
    FALDA_ALTO = 0.22           # m: la parte de abajo de la camiseta que se apoya en el regazo al sentarse
    FALDA_SUBE = 0.75           # cuánto se recoge la parte de abajo (0 = cuelga, 1 = toda hasta la cintura)
    FALDA_SOBRE = -0.02         # m respecto a la cadera: lo de abajo de aquí se recoge
    MUSLO_DESDE, MUSLO_HASTA = 35.0, 85.0      # grados de muslo levantado: desde aquí empieza / del todo en el regazo

    def _falda(s):
        """Para la parte de abajo de la camiseta: pesos como la piel más cercana (cadera y muslos) y cuánto es falda."""
        if getattr(s, '_falda_d', None) is None:
            from scipy.spatial import cKDTree
            nJ = len(s.nombres)
            piel = [i for i, p in enumerate(s.prims) if p['mname'].startswith(('Std_Skin_Body', 'Std_Skin_Leg'))]
            X = np.concatenate([s.prims[i]['P'] for i in piel])
            D = np.concatenate([_denso(s.Jn[i], s.Wn[i], nJ) for i in piel])
            arbol = cKDTree(X); out = {}
            for k, p in enumerate(s.prims):
                if not p['mname'].startswith('Poloche'):
                    continue
                P = p['P']; y = P[:, 1]; y0 = y.min()
                f = 1 - np.clip((y - y0) / (s.FALDA_ALTO / s.base.scale), 0, 1); f = f * f * (3 - 2 * f)
                d, ix = arbol.query(P, k=8); w = 1.0 / np.maximum(d, 1e-4) ** 2; w /= w.sum(1, keepdims=True)
                Wd = np.einsum('vk,vkj->vj', w, D[ix]); top = np.argsort(-Wd, axis=1)[:, :4]
                Wt = np.take_along_axis(Wd, top, 1); Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
                lados = {L: np.isin(top, [s.idx[L + 'Leg'], s.idx[L + 'Shin']]) for L in ('Left', 'Right')}
                pierna = {L: (Wt * lados[L]).sum(1) for L in lados}
                out[k] = dict(J=top, W=Wt, f=f, pierna=pierna)
            s._falda_d = out
        return s._falda_d

    def _muslo_arriba(s, Pm):
        """Grados que cada muslo está levantado hacia adelante respecto al tronco (parado ~0, sentado ~90)."""
        ix = s.idx; abajo = Pm[ix['Hips']] - Pm[ix['Spine2']]; abajo /= np.linalg.norm(abajo) + 1e-12
        out = {}
        for L in ('Left', 'Right'):
            m = Pm[ix[L + 'Shin']] - Pm[ix[L + 'Leg']]; m /= np.linalg.norm(m) + 1e-12
            out[L] = np.degrees(np.arccos(np.clip(m @ abajo, -1, 1)))
        return out

    def pose_mats(s, G, Pm, b, apply=True):
        res = super().pose_mats(G, Pm, b, apply)
        if not (s.CAMISETA_SENTADO and apply):
            return res
        # 1. sentado o en cuclillas: la parte de abajo de la camiseta se recoge hacia la cintura (como la tela de verdad,
        #    que se arruga sobre la barriga) en vez de colgar hasta el piso y doblarse debajo de los muslos
        ang = s._muslo_arriba(Pm)
        cuanto = max(float(np.clip((a_ - s.MUSLO_DESDE) / (s.MUSLO_HASTA - s.MUSLO_DESDE), 0, 1)) for a_ in ang.values())
        cuanto = cuanto * cuanto * (3 - 2 * cuanto)
        if cuanto > 1e-3:
            Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
            fin = lambda x: ((Cz @ x) - [0, 0, s.base.zoff]) * s.base.scale
            cad = fin(Pm[s.idx['Hips']]); up = fin(Pm[s.idx['Spine2']]) - cad; up /= np.linalg.norm(up) + 1e-12
            for k, d in s._falda().items():
                p = s.prims[k]; V, N = res[k]
                fv = d['f'][p['idx']]
                h = (V - cad) @ up
                baja = (fv > 0) & (h < s.FALDA_SOBRE)
                if baja.any():
                    k_ = s.FALDA_SUBE * cuanto * fv[baja]
                    h2 = s.FALDA_SOBRE + (h[baja] - s.FALDA_SOBRE) * (1 - k_)
                    V = V.copy(); V[baja] += up * (h2 - h[baja])[:, None]
                    res[k] = (V, N)
        from scipy.spatial import cKDTree
        kp = [i for i, p in enumerate(s.prims) if p['mname'] == 'Pants']
        if not kp:
            return res
        PV, PN = res[kp[0]]
        PV = PV[::3]; PN = PN[::3]                                   # (cada triángulo repite sus puntos)
        arbol = cKDTree(PV)
        for k, p in enumerate(s.prims):
            if not p['mname'].startswith('Poloche'):
                continue
            V, N = res[k]; V = V.copy(); cambio = False
            d, ix = arbol.query(V, k=4, distance_upper_bound=s.TELA_BUSCA)
            ok = np.isfinite(d[:, 0])
            if ok.any():
                ii = np.where(ok)[0]; jj = ix[ii]; dd = d[ii]
                valid = np.isfinite(dd); jj = np.where(valid, jj, jj[:, :1])
                w = np.where(valid, 1.0 / np.maximum(dd, 1e-4), 0); w /= w.sum(1, keepdims=True)
                q = np.einsum('vk,vkj->vj', w, PV[jj]); nq = np.einsum('vk,vkj->vj', w, PN[jj])
                nq /= np.linalg.norm(nq, axis=1, keepdims=True) + 1e-9
                prof = ((V[ii] - q) * nq).sum(1)                         # < 0: la tela quedó dentro del pantalón
                den = prof < s.MUSLO_HOLGURA * 0.5
                # solo donde el muslo está levantado hacia la camiseta (no en la cintura de siempre)
                if den.any():
                    sel = ii[den]
                    V[sel] = V[sel] + nq[den] * (s.MUSLO_HOLGURA - prof[den])[:, None]; cambio = True
            baj = V[:, 2] < s.TELA_PISO + getattr(s, 'PISO_Z', 0.0)
            if baj.any():
                V[baj, 2] = s.TELA_PISO + getattr(s, 'PISO_Z', 0.0); cambio = True
            if cambio:
                res[k] = (V, N)
        return res


class MaxV17(_Piso, _CamisetaSentado, _PantalonComoLaPiel, MaxV16):
    """Max V17 sin cara viva."""


class MaxCaraV17(_Piso, _CamisetaSentado, _PantalonComoLaPiel, MaxCaraV16):
    """La que se usa desde el ZIP 43: Max V16 + piso + ropa al sentarse."""
