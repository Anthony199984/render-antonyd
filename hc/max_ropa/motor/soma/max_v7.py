"""Max v7 (ZIP_29, archivo NUEVO: no cambia nada de lo que ya hay). Es MaxV6 / MaxCaraV6 con la PIERNA IZQUIERDA arreglada.

QUÉ ARREGLA: igual que el brazo derecho (ZIP 27/28), los BVH de Kimodo traen la pierna izquierda (muslo, canilla, pie
y dedos) girada ~120° sobre el eje de cada hueso respecto al reposo (la derecha viene casi en 0). Eso retorcía la piel
y el pantalón del muslo izquierdo y doblaba el tenis izquierdo respecto a la canilla.
Medido en 2 clips (escena de la botella y caminar), giro respecto al hueso de arriba:
    antes:  muslo -131°, pie -77° (y 113° de doblez)      ahora: muslo -11°, pie -2° (y 9° de doblez)
    pierna derecha (referencia): muslo 10°, pie 3° (y 9° de doblez)
Girar un hueso sobre su propio eje NO mueve ninguna articulación: cadera, rodilla, tobillo y pies (y el bloqueo de
pies en el piso) quedan exactamente en el mismo lugar.

Uso (igual que MaxCaraV6 / MaxV6, solo cambia el nombre):
    from soma.max_v7 import MaxCaraV7, MaxV7
"""
import numpy as np

from .max_v6 import MaxV6, MaxCaraV6
from .cara import PistaCara


class _PiernaIzquierda:
    PIERNA = {'LeftLeg': -121.0, 'LeftShin': -121.0, 'LeftFoot': -121.0, 'LeftToeBase': -121.0}
    _HIJO_PIERNA = {'LeftLeg': 'LeftShin', 'LeftShin': 'LeftFoot', 'LeftFoot': 'LeftToeBase', 'LeftToeBase': 'LeftToeEnd'}

    def _bind(s, clip):
        b = super()._bind(clip)
        clave = ('v7', id(b), tuple(sorted(s.PIERNA.items())))
        c = s.__dict__.setdefault('_cache_v7', {})
        if clave in c:
            return c[clave][1]
        A = b['A'].copy(); P = b['P']; ix = s.idx
        C = s._correccion()
        for n, gr in s.PIERNA.items():
            if not gr:
                continue
            j, h = ix[n], ix[s._HIJO_PIERNA[n]]
            e = C[j] @ (P[h] - P[j]); e = e / np.linalg.norm(e)
            g = np.radians(gr)
            K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            A[j] = (np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K) @ A[j]
        off = np.zeros_like(P)
        for j, p in enumerate(s.padres):
            if p >= 0:
                off[j] = A[p].T @ (P[j] - P[p])
        ibm = b['ibm'].copy()
        for n in s.PIERNA:
            j = ix[n]
            M = np.eye(4); M[:3, :3] = A[j]; M[:3, 3] = P[j]
            ibm[j] = np.linalg.inv(M)
        nb = dict(b); nb.update(A=A, off=off, ibm=ibm)
        c[clave] = (b, nb)
        return nb


class _ManoNoAtraviesaCara:
    """Si los dedos o la palma entran en la cara o la cabeza (gestos de mano en la cara, rascarse, pensar...), la mano
    se aparta lo justo para quedar TOCANDO la piel (IK del brazo; el giro de la mano y los dedos no cambia).
    Solo actúa cuando hay contacto con la cabeza; en el resto de los cuadros no toca nada (agarres incluidos)."""
    MANO_CARA = True
    RADIO_DEDO, RADIO_PALMA = 0.008, 0.016

    def _cabeza_y_cuello(s):
        """Vértices de la cara, la cabeza y el CUELLO (piel), con sus pesos: se ponen en pose en cada cuadro."""
        if not hasattr(s, '_cyc'):
            ix = s.idx
            cab = [ix[n] for n in ('Head', 'Neck1', 'Neck2', 'Jaw') if n in ix]
            P, N, J, W = [], [], [], []
            for k, p in enumerate(s.prims):
                if not p['mname'].startswith(('Std_Skin_Head', 'Std_Skin_Body')):
                    continue
                Jn, Wn = s.Jn[k], s.Wn[k]
                peso = (Wn * np.isin(Jn, cab)).sum(1)
                v = np.unique(p.get('idx_orig', p['idx']))
                v = v[peso[v] > (0.6 if p['mname'].startswith('Std_Skin_Body') else 0.3)]
                P.append(p['P'][v]); N.append(p['N'][v]); J.append(Jn[v]); W.append(Wn[v])
            kmax = max(x.shape[1] for x in J)
            J = np.vstack([np.pad(x, ((0, 0), (0, kmax - x.shape[1]))) for x in J])
            W = np.vstack([np.pad(x, ((0, 0), (0, kmax - x.shape[1]))) for x in W])
            s._cyc = (np.vstack(P), np.vstack(N), J, W)
        return s._cyc

    def _superficie(s, G, Pm, b):
        from scipy.spatial import cKDTree
        from .rig import _M
        P, N, J, W = s._cabeza_y_cuello()
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        M = np.einsum('vk,vkij->vij', W, K[J])
        V = np.einsum('vij,vj->vi', M[:, :3, :3], P) + M[:, :3, 3]
        Nn = np.einsum('vij,vj->vi', M[:, :3, :3], N); Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
        return cKDTree(V), V, Nn

    def _puntos_mano(s, L):
        if not hasattr(s, '_pm'):
            s._pm = {}
        if L not in s._pm:
            ix = s.idx
            dedos = [ix[n] for n in s.nombres if n.startswith(L + 'Hand') and n != L + 'Hand']
            s._pm[L] = (ix[L + 'Hand'], ix[L + 'HandMiddle1'], dedos)
        return s._pm[L]

    def _hundimiento(s, G, Pm, b, L, sup=None):
        arbol, V, Nn = sup if sup is not None else s._superficie(G, Pm, b)
        iH, iM, dedos = s._puntos_mano(L)
        pts = np.array([Pm[j] for j in dedos] + [Pm[iH] * 0.4 + Pm[iM] * 0.6, Pm[iH]])
        rad = np.array([s.RADIO_DEDO] * len(dedos) + [s.RADIO_PALMA, s.RADIO_PALMA])
        d, j = arbol.query(pts)
        sd = np.einsum('ij,ij->i', pts - V[j], Nn[j])
        hondo = rad - sd
        hondo[d > 0.06] = -1
        k = int(np.argmax(hondo))
        dentro = hondo > 0
        if not dentro.any():
            return hondo[k], Nn[j[k]]
        n = (Nn[j[dentro]] * hondo[dentro, None]).sum(0)
        return hondo[k], n / (np.linalg.norm(n) + 1e-12)

    def _ik_mano(s, G, Pm, b, L, T):
        from .rig import _swing
        ix = s.idx
        iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
        A_, E, H = Pm[iA], Pm[iF], Pm[iH]
        l1, l2 = np.linalg.norm(E - A_), np.linalg.norm(H - E)
        dv = T - A_; d = min(np.linalg.norm(dv), (l1 + l2) * .9995); e = dv / np.linalg.norm(dv)
        x = (l1 * l1 - l2 * l2 + d * d) / (2 * d); hh = np.sqrt(max(l1 * l1 - x * x, 0.0))
        pol = (E - A_) - e * ((E - A_) @ e)
        if np.linalg.norm(pol) < 1e-6:
            return
        pol = pol / np.linalg.norm(pol)
        E2 = A_ + x * e + hh * pol; H2 = A_ + d * e
        R1 = _swing(E - A_, E2 - A_); G[iA] = R1 @ G[iA]
        Gf = R1 @ G[iF]; H1 = E2 + Gf @ b['off'][iH]; G[iF] = _swing(H1 - E2, H2 - E2) @ Gf
        if not hasattr(s, '_desc_brazo'):
            s._desc_brazo = {}
        if L not in s._desc_brazo:
            dd = [iA]
            for jj in range(len(s.nombres)):
                if jj not in dd and s.padres[jj] in dd: dd.append(jj)
            s._desc_brazo[L] = sorted(dd)
        for jj in s._desc_brazo[L]:
            if jj != iA:
                Pm[jj] = Pm[s.padres[jj]] + G[s.padres[jj]] @ b['off'][jj]

    # cuánto puede cambiar la corrección de un cuadro al siguiente (si no, el brazo "tiembla" al tocar y soltar)
    GIRO_CODO_MAX = 64.0          # grados como máximo que se gira el codo alrededor del eje hombro-muñeca
    VEL_GIRO_CODO = 150.0         # grados por segundo
    VEL_EMPUJE = 0.30             # m/s
    RADIOS_CODO = (0.040, 0.036)  # codo, brazo cerca del codo

    def _puntos_brazo(s, S, E, H):
        return np.array([E, S + (E - S) * 0.8, E + (H - E) * 0.35, E + (H - E) * 0.7]), \
            np.array([s.RADIOS_CODO[0], s.RADIOS_CODO[1], s.RADIOS_BRAZO[2], s.RADIOS_BRAZO[1]])

    def _pen_brazo(s, sup, L, S, E, H):
        arbol, V, Nn = sup[L]
        pts, rad = s._puntos_brazo(S, E, H)
        d, j = arbol.query(pts)
        sd = np.einsum('ij,ij->i', pts - V[j], Nn[j])
        h = rad - sd; h[d > 0.08] = -1
        return float(h.max())

    @staticmethod
    def _rot_eje(e, g):
        e = e / (np.linalg.norm(e) + 1e-12); K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        return np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K

    # BRAZOS AFUERA: el codo y el brazo nunca quedan pegados ni metidos en el tronco: siempre una holgura
    HOLGURA_BRAZO = 0.015         # m de aire entre el brazo y la camiseta (además del grosor del brazo)
    ABRIR_MAX = 30.0              # grados como máximo que se abre el brazo hacia afuera (mano libre)
    VEL_ABRIR = 120.0             # grados por segundo

    def _falta_holgura(s, sup, S, E, H):
        """Cuánto le falta al brazo (codo, brazo, antebrazo) para tener HOLGURA_BRAZO de aire con el tronco (m)."""
        arbol, V, Nn = sup['tronco']
        pts = np.array([E, S + (E - S) * 0.55, S + (E - S) * 0.8, E + (H - E) * 0.35, E + (H - E) * 0.7])
        rad = np.array([0.040, 0.045, 0.042, 0.035, 0.032]) + s.HOLGURA_BRAZO
        d, j = arbol.query(pts)
        sd = np.einsum('ij,ij->i', pts - V[j], Nn[j])
        h = rad - sd; h[d > 0.12] = -1
        return float(h.max())

    def _abrir_brazo(s, G, Pm, sup, L):
        """Grados que hay que abrir el brazo (hacia afuera, alrededor del hombro) para que no toque el tronco."""
        ix = s.idx
        iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
        S, E, H = Pm[iA], Pm[iF], Pm[iH]
        if s._falta_holgura(sup, S, E, H) <= 0:
            return 0.0, None
        fwd = G[ix['Chest']] @ s._fwd_pecho()
        eje = s._eje_abrir(G, Pm, L)
        for a in np.arange(3.0, s.ABRIR_MAX + 0.1, 3.0):
            R = s._rot_eje(eje, np.radians(a))
            if s._falta_holgura(sup, S, S + R @ (E - S), S + R @ (H - S)) <= 0:
                return float(a), eje
        return s.ABRIR_MAX, eje

    def _fwd_pecho(s):
        if not hasattr(s, '_fp'):
            b = s._ultimo_b
            # "adelante" del pecho de Max en su reposo: +Z del glTF, pasado al marco del hueso
            s._fp = b['A'][s.idx['Chest']].T @ np.array([0, 0, 1.0])
        return s._fp

    def _eje_abrir(s, G, Pm, L):
        """Eje alrededor del hombro que lleva el brazo hacia AFUERA (lejos de la columna)."""
        ix = s.idx
        S, E = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']]
        fwd = G[ix['Chest']] @ s._fwd_pecho()
        eje = fwd / (np.linalg.norm(fwd) + 1e-12)
        col = Pm[ix['Chest']]
        lat = S - col; lat -= eje * (lat @ eje); lat /= np.linalg.norm(lat) + 1e-12
        R = s._rot_eje(eje, np.radians(5))
        if ((S + R @ (E - S)) - E) @ lat < 0:
            eje = -eje
        return eje

    def _giro_codo(s, G, Pm, sup, L):
        """Ángulo (grados) que hay que girar el codo alrededor del eje hombro-muñeca para que codo y antebrazo no
        entren en el cuerpo. La mano NO se mueve (sirve también cuando agarra)."""
        ix = s.idx
        S, E, H = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']], Pm[ix[L + 'Hand']]
        if max(s._pen_brazo(sup, L, S, E, H), s._falta_holgura(sup, S, E, H)) <= 0.002:
            return 0.0
        mejor = None
        for a in sorted(np.arange(-s.GIRO_CODO_MAX, s.GIRO_CODO_MAX + 1, 8.0), key=abs):
            R = s._rot_eje(H - S, np.radians(a))
            E2 = S + R @ (E - S)
            p = max(s._pen_brazo(sup, L, S, E2, H), s._falta_holgura(sup, S, E2, H))
            if p <= 0.002:
                return float(a)
            if mejor is None or p < mejor[0] - 1e-4:
                mejor = (p, float(a))
        return mejor[1]

    def _aplicar_giro_codo(s, G, Pm, b, L, a):
        if abs(a) < 1e-3:
            return
        ix = s.idx
        iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
        S, H = Pm[iA], Pm[iH]
        R = s._rot_eje(H - S, np.radians(a))
        Gh = G[iH].copy()
        G[iA] = R @ G[iA]; G[iF] = R @ G[iF]
        Pm[iF] = S + R @ (Pm[iF] - S)
        G[iH] = Gh                                       # la mano queda igual (posición y giro)

    def _empuje(s, G, Pm, b, L, sup_cara, sup_cuerpo):
        """Cuánto hay que mover la mano (vector) para que no entre en la cara/cuello ni en el cuerpo (en copias)."""
        Gc, Pc = G.copy(), Pm.copy()
        iH = s.idx[L + 'Hand']; H0 = Pc[iH].copy()
        if sup_cara is not None:
            for it in range(16):
                h, n = s._hundimiento(Gc, Pc, b, L, sup_cara)
                if h <= 0.0005: break
                s.mano_cara_mm = max(s.mano_cara_mm, h * 1000)
                s._ik_mano(Gc, Pc, b, L, Pc[iH] + n * (h * (1.0 if it < 6 else 1.6) + 0.001))
        if sup_cuerpo is not None:
            for it in range(12):
                h, n = s._hundimiento_cuerpo(Gc, Pc, b, L, sup_cuerpo)
                if h <= 0.0005: break
                s.mano_cuerpo_mm = max(s.mano_cuerpo_mm, h * 1000)
                s._ik_mano(Gc, Pc, b, L, Pc[iH] + n * (h * (1.0 if it < 5 else 1.6) + 0.001))
        return Pc[iH] - H0

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        s.mano_cara_mm = 0.0; s.mano_cuerpo_mm = 0.0
        est = s.__dict__.setdefault('_est_mano', {})
        seguido = est.get('clip') == id(clip) and 0 < t - est.get('t', -9.0) <= 0.1
        dt = (t - est['t']) if seguido else None
        ih = s.idx['Head']
        sup_cuerpo = s._superficie_cuerpo(G, Pm, b) if (s.MANO_CUERPO or s.MANO_CARA) else None
        s._ultimo_b = b
        for L in ('Left', 'Right'):
            libre = getattr(s, '_dedos_libres', {}).get(L, 1.0) >= 0.5
            # 0) mano libre: el brazo se abre hacia afuera lo justo para no tocar el tronco (codos afuera)
            ab = 0.0; eje = None
            if libre and s.MANO_CUERPO:
                ab, eje = s._abrir_brazo(G, Pm, sup_cuerpo, L)
            if dt is not None:
                prev = est.get(('abrir', L), 0.0); paso = s.VEL_ABRIR * dt
                ab = prev + float(np.clip(ab - prev, -paso, paso))
            est[('abrir', L)] = ab
            if ab > 0.05:
                eje = s._eje_abrir(G, Pm, L) if eje is None else eje
                iA = s.idx[L + 'Arm']; S = Pm[iA].copy(); R = s._rot_eje(eje, np.radians(ab))
                if not hasattr(s, '_sub_brazo'):
                    s._sub_brazo = {}
                if L not in s._sub_brazo:
                    dd = [iA]
                    for jj in range(len(s.nombres)):
                        if jj not in dd and s.padres[jj] in dd: dd.append(jj)
                    s._sub_brazo[L] = dd
                for jj in s._sub_brazo[L]:
                    G[jj] = R @ G[jj]; Pm[jj] = S + R @ (Pm[jj] - S)
            # 1) codo fuera del cuerpo girando el brazo sobre el eje hombro-muñeca (también si agarra)
            a = s._giro_codo(G, Pm, sup_cuerpo, L) if s.MANO_CUERPO else 0.0
            if dt is not None:
                prev = est.get(('codo', L), 0.0); paso = s.VEL_GIRO_CODO * dt
                a = prev + float(np.clip(a - prev, -paso, paso))
            est[('codo', L)] = a
            s._aplicar_giro_codo(G, Pm, b, L, a)
            # 2) mano libre: fuera de la cara, el cuello y el cuerpo
            d = np.zeros(3)
            if libre:
                sc = None
                if s.MANO_CARA and np.linalg.norm(Pm[s.idx[L + 'Hand']] - Pm[ih]) <= 0.35:
                    sc = s._superficie(G, Pm, b)
                d = s._empuje(G, Pm, b, L, sc, sup_cuerpo if s.MANO_CUERPO else None)
            if dt is not None:
                prev = est.get(('mano', L), np.zeros(3)); paso = s.VEL_EMPUJE * dt
                dd = d - prev; n = np.linalg.norm(dd)
                if n > paso:
                    d = prev + dd * (paso / n)
            est[('mano', L)] = d
            if np.linalg.norm(d) > 1e-4:
                s._ik_mano(G, Pm, b, L, Pm[s.idx[L + 'Hand']] + d)
        est['clip'] = id(clip); est['t'] = float(t)
        return G, Pm, b

    # ---- mano y antebrazo contra el CUERPO (la camiseta, o la piel sin camisa), cuando la mano no agarra nada
    MANO_CUERPO = True
    RADIOS_BRAZO = (0.025, 0.032, 0.036)    # muñeca, antebrazo a 2/3, antebrazo a 1/3 (desde la muñeca)

    def _cuerpo_region(s):
        if not hasattr(s, '_creg'):
            ix = s.idx
            nombres = ('Poloche_Algodon_Blanco',) if any(p['mname'].startswith('Poloche_Algodon') for p in s.prims) \
                else ('Std_Skin_Body',)
            P, N, J, W, lado = [], [], [], [], []
            brazoL = [ix['Left' + n] for n in ('Arm', 'ForeArm')]; brazoR = [ix['Right' + n] for n in ('Arm', 'ForeArm')]
            for k, p in enumerate(s.prims):
                if not p['mname'].startswith(nombres):
                    continue
                Jn, Wn = s.Jn[k], s.Wn[k]
                v = np.unique(p.get('idx_orig', p['idx']))
                P.append(p['P'][v]); N.append(p['N'][v]); J.append(Jn[v]); W.append(Wn[v])
                lado.append(np.stack([(Wn[v] * np.isin(Jn[v], brazoL)).sum(1), (Wn[v] * np.isin(Jn[v], brazoR)).sum(1)], 1))
            kmax = max(x.shape[1] for x in J)
            J = np.vstack([np.pad(x, ((0, 0), (0, kmax - x.shape[1]))) for x in J])
            W = np.vstack([np.pad(x, ((0, 0), (0, kmax - x.shape[1]))) for x in W])
            s._creg = (np.vstack(P), np.vstack(N), J, W, np.vstack(lado))
        return s._creg

    def _superficie_cuerpo(s, G, Pm, b):
        from scipy.spatial import cKDTree
        from .rig import _M
        P, N, J, W, lado = s._cuerpo_region()
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        M = np.einsum('vk,vkij->vij', W, K[J])
        V = np.einsum('vij,vj->vi', M[:, :3, :3], P) + M[:, :3, 3]
        Nn = np.einsum('vij,vj->vi', M[:, :3, :3], N); Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
        out = {}
        for i, L in enumerate(('Left', 'Right')):
            m = lado[:, i] < 0.3                  # sin la manga de ese mismo brazo
            out[L] = (cKDTree(V[m]), V[m], Nn[m])
        m = (lado[:, 0] < 0.15) & (lado[:, 1] < 0.15)  # solo el tronco (sin ninguna manga)
        out['tronco'] = (cKDTree(V[m]), V[m], Nn[m])
        return out

    def _hundimiento_cuerpo(s, G, Pm, b, L, sup):
        arbol, V, Nn = sup[L]
        ix = s.idx
        iH, iM, dedos = s._puntos_mano(L)
        iF = ix[L + 'ForeArm']
        pts = [Pm[j] for j in dedos] + [Pm[iH] * 0.4 + Pm[iM] * 0.6, Pm[iH],
                                        Pm[iH] * 0.67 + Pm[iF] * 0.33, Pm[iH] * 0.33 + Pm[iF] * 0.67]
        rad = [s.RADIO_DEDO] * len(dedos) + [s.RADIO_PALMA] + list(s.RADIOS_BRAZO)
        pts = np.array(pts); rad = np.array(rad)
        d, j = arbol.query(pts)
        sd = np.einsum('ij,ij->i', pts - V[j], Nn[j])
        hondo = rad - sd
        hondo[d > 0.08] = -1
        k = int(np.argmax(hondo))
        dentro = hondo > 0
        if not dentro.any():
            return hondo[k], Nn[j[k]]
        n = (Nn[j[dentro]] * hondo[dentro, None]).sum(0)
        return hondo[k], n / (np.linalg.norm(n) + 1e-12)


# ------------------------------------------------------------------------------------------------ peso de los objetos
def _suave01(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def esfuerzo_de_peso(kg):
    """0 = sin esfuerzo (hasta 2 kg), 1 = esfuerzo máximo (20 kg o más)."""
    return float(np.clip((float(kg) - 2.0) / 18.0, 0.0, 1.0))


def cargas_de_escena(carpeta, escena_json):
    """Lee los agarres de la escena y el "peso" (kg) de cada objeto del JSON -> [(t_agarra, t_suelta, esfuerzo, mano)]."""
    import json
    from pathlib import Path
    carpeta = Path(carpeta)
    inf = next(carpeta.glob("*_informe.json"), None)
    if inf is None or escena_json is None:
        return []
    informe = json.loads(inf.read_text(encoding="utf-8"))
    esc = json.loads(Path(escena_json).read_text(encoding="utf-8"))
    objetos = esc.get("objetos", {})
    agarre_json = {a["agarre"].get("objeto"): a["agarre"] for a in esc.get("acciones", []) if "agarre" in a}
    out = []
    for a in informe.get("agarres", []):
        o = a.get("objeto"); kg = objetos.get(o, {}).get("peso")
        if kg is None or a.get("segundo") is None:
            continue
        e = esfuerzo_de_peso(kg)
        if e <= 0:
            continue
        t0 = float(a["segundo"])
        dur = a.get("sostener") or agarre_json.get(o, {}).get("sostener") or (a.get("cuadros_agarrando", 60) / 30.0)
        out.append((t0, t0 + float(dur), e, a.get("mano", "der")))
    return out


class PistaCaraV7(PistaCara):
    """PistaCara + cara de ESFUERZO cuando el objeto agarrado pesa ("peso" en kg en los objetos del JSON)."""

    @classmethod
    def desde_escena(cls, carpeta, escena_json=None, semilla=7):
        p = PistaCara.desde_escena(carpeta, escena_json, semilla)
        p.__class__ = cls
        p.cargas = cargas_de_escena(carpeta, escena_json)
        tr = []
        for t0, t1, ex, inten, hab in p.tramos:
            if ex == "concentrado":
                for c0, c1, e, _ in p.cargas:
                    if c0 < t1 and c1 > t0:
                        inten = inten * (1 - e)
            tr.append((t0, t1, ex, inten, hab))
        for c0, c1, e, _ in p.cargas:
            tr.append((c0 - 0.15, c1, "esfuerzo", 0.35 + 0.65 * e, False))
        p.tramos = sorted(tr)
        return p


class _PesoDelObjeto:
    """Con un objeto pesado en la mano el cuerpo compensa: el tronco se echa hacia el lado contrario a la carga (atrás si
    la lleva adelante, al otro lado si la lleva al costado) y el hombro de esa mano baja. Las manos siguen exactamente
    donde las pone agarres_ik (el IK del brazo se hace después)."""
    INCLINA_MAX = 12.0       # grados de tronco con esfuerzo 1
    HOMBRO_MAX = 7.0         # grados que baja la clavícula de la mano que carga
    ENTRA_S = 0.35

    def _carga(s, t):
        p = getattr(s, 'pista', None)
        mejor = (0.0, None)
        for c0, c1, e, mano in getattr(p, 'cargas', []) or []:
            a = _suave01((t - c0 + s.ENTRA_S) / s.ENTRA_S) * _suave01((c1 + s.ENTRA_S - t) / s.ENTRA_S) * e
            if a > mejor[0]:
                mejor = (a, mano)
        return mejor

    def _girar_sub(s, G, Pm, raiz, R, piv):
        if not hasattr(s, '_subarbol'):
            s._subarbol = {}
        if raiz not in s._subarbol:
            d = [raiz]
            for j in range(len(s.nombres)):
                if j not in d and s.padres[j] in d: d.append(j)
            s._subarbol[raiz] = d
        for j in s._subarbol[raiz]:
            Pm[j] = piv + R @ (Pm[j] - piv); G[j] = R @ G[j]

    @staticmethod
    def _rot(eje, g):
        e = eje / np.linalg.norm(eje); K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        return np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K

    def _manos_como_antes(s, clip, t, bucle, G, Pm, b):
        a, mano = s._carga(float(t))
        s.esfuerzo_actual = a
        if a > 1e-3:
            ix = s.idx; L = 'Right' if mano == 'der' else 'Left'
            P0 = s._posiciones_maxsoma(clip, t, bucle)
            arriba = np.array([0, 1.0, 0])
            v = P0[ix[L + 'Hand']] - Pm[ix['Hips']]; v[1] = 0
            if np.linalg.norm(v) > 1e-3:
                eje = np.cross(arriba, -v)             # inclinar hacia el lado contrario a la carga
                g = np.radians(s.INCLINA_MAX * a) / 3
                for n in ('Spine1', 'Spine2', 'Chest'):
                    s._girar_sub(G, Pm, ix[n], s._rot(eje, g), Pm[ix[n]].copy())
            # bajar el hombro de la mano que carga
            iS, iA = ix[L + 'Shoulder'], ix[L + 'Arm']
            d = Pm[iA] - Pm[iS]
            eje = np.cross(d, -arriba)
            if np.linalg.norm(eje) > 1e-6:
                s._girar_sub(G, Pm, iS, s._rot(eje, np.radians(s.HOMBRO_MAX * a)), Pm[iS].copy())
        return super()._manos_como_antes(clip, t, bucle, G, Pm, b)


# ------------------------------------------------------------------------------------------------ codos sin quebrarse
def _dq_de_matrices(K):
    """Matrices rígidas 4x4 (n) -> cuaterniones duales (real (n,4) wxyz, dual (n,4))."""
    from scipy.spatial.transform import Rotation as Ro
    q = Ro.from_matrix(K[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    t = K[:, :3, 3]
    w, x, y, z = q.T
    tx, ty, tz = t.T
    d = 0.5 * np.stack([-tx * x - ty * y - tz * z,
                        tx * w + ty * z - tz * y,
                        -tx * z + ty * w + tz * x,
                        tx * y - ty * x + tz * w], 1)
    return q, d


def _matrices_de_dq(q, d):
    n = np.linalg.norm(q, axis=1, keepdims=True); q = q / n; d = d / n
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    dw, dx, dy, dz = d.T
    t = 2 * np.stack([-dw * x + dx * w - dy * z + dz * y,
                      -dw * y + dx * z + dy * w - dz * x,
                      -dw * z - dx * y + dy * x + dz * w], 1)
    return R, t


class _PielSinQuebrarse:
    """Codos, muñecas y hombros: la piel y la camiseta se doblan con "cuaterniones duales" en vez de la mezcla lineal
    de matrices. Con la mezcla lineal, al doblar mucho el codo (o girar la muñeca) la piel se hunde y hace una punta
    o un pliegue oscuro ("se quiebra"); así el brazo conserva su grosor. Solo cambia cómo la piel sigue a los huesos:
    los huesos (y los agarres) no se mueven. Se usa en la piel de codos, antebrazos y muñecas."""
    PIEL_DQ = True

    def _mascara_dq(s, k):
        c = s.__dict__.setdefault('_mdq', {})
        if k not in c:
            ix = s.idx
            # solo piel de codo, antebrazo y muñeca (el hombro y las mangas siguen con la mezcla lineal: con duales la
            # manga se inflaba al subir el brazo)
            brazo = [ix[L + n] for L in ('Left', 'Right') for n in ('ForeArm', 'Hand')]
            Jn, Wn = s.Jn[k], s.Wn[k]
            piel = s.prims[k]['mname'].startswith(('Std_Skin_Arm', 'Std_Nails'))
            c[k] = np.clip((Wn * np.isin(Jn, brazo)).sum(1) * 2.0, 0, 1) * (1.0 if piel else 0.0)
        return c[k]

    def _mezcla(s, k, K, KQ, KD):
        Jn, Wn = s.Jn[k], s.Wn[k]
        Mv = np.einsum('vk,vkij->vij', Wn, K[Jn])
        if not s.PIEL_DQ:
            return Mv[:, :3, :3], Mv[:, :3, 3]
        a = s._mascara_dq(k)
        sel = a > 0
        R, t = Mv[:, :3, :3].copy(), Mv[:, :3, 3].copy()
        if sel.any():
            J, W = Jn[sel], Wn[sel]
            q = KQ[J]; d = KD[J]
            sg = np.sign(np.einsum('vkc,vc->vk', q, q[np.arange(len(J)), np.argmax(W, 1)]))
            sg[sg == 0] = 1
            qb = np.einsum('vk,vkc->vc', W * sg, q); db = np.einsum('vk,vkc->vc', W * sg, d)
            Rd, td = _matrices_de_dq(qb, db)
            aa = a[sel][:, None]
            R[sel] = Rd * aa[:, :, None] + R[sel] * (1 - aa[:, :, None])
            t[sel] = td * aa + t[sel] * (1 - aa)
        return R, t

    # "delta mush" en codos y muñecas: después de doblar, la piel de esa zona se alisa y se le devuelve su relieve
    # original (medido en reposo, en el marco local de cada punto). Quita la punta y la grieta del codo doblado.
    ALISAR_CODO = True
    ALISAR_RADIO = 0.11      # m alrededor del codo (y 0,07 alrededor de la muñeca)
    ALISAR_PASOS = 12

    def _prep_alisar(s, k):
        c = s.__dict__.setdefault('_alis', {})
        if k in c:
            return c[k]
        from scipy.sparse import coo_matrix, diags
        p = s.prims[k]
        P0 = p.get('Pm', p['P']).astype(float)
        key = np.round(p['P'] / 1e-5).astype(np.int64)
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
        n = len(rep)
        F = inv[p['idx_orig'] if 'idx_orig' in p else p['idx']].reshape(-1, 3)
        i = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
        j = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
        A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr(); A.data[:] = 1
        deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
        S = diags(1 / deg) @ A
        # zona: alrededor de codos y muñecas (en reposo), con caída suave
        Pw = P0[rep]
        a = np.zeros(n)
        for L in ('Left', 'Right'):
            for hueso, r in (('ForeArm', s.ALISAR_RADIO), ('Hand', 0.07)):
                d = np.linalg.norm(Pw - s.pA[L + hueso], axis=1)
                a = np.maximum(a, np.clip(1.5 - 1.5 * d / r, 0, 1))
        a = np.clip(a, 0, 1)
        # vecino de referencia para el marco local
        vec = np.full(n, -1); vec[F[:, 0]] = F[:, 1]; vec[F[:, 1]] = F[:, 2]; vec[F[:, 2]] = F[:, 0]
        def marco(X):
            Sm = X.copy()
            for _ in range(s.ALISAR_PASOS): Sm = 0.5 * Sm + 0.5 * (S @ Sm)
            fn = np.cross(Sm[F[:, 1]] - Sm[F[:, 0]], Sm[F[:, 2]] - Sm[F[:, 0]])
            N = np.zeros_like(Sm); np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            T = Sm[vec] - Sm; T -= N * np.einsum('ij,ij->i', T, N)[:, None]
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            B = np.cross(N, T)
            return Sm, np.stack([T, B, N], 2)          # columnas: T, B, N
        Sm0, M0 = marco(Pw)
        delta = np.einsum('nji,nj->ni', M0, Pw - Sm0)  # coordenadas locales del relieve
        sel = a > 0
        c[k] = dict(rep=rep, inv=inv, a=a, sel=sel, delta=delta, marco=marco, F=F)
        return c[k]

    def _alisar(s, k, V):
        d = s._prep_alisar(k)
        if not d['sel'].any():
            return V, None
        Vw = V[d['rep']]
        Sm, M = d['marco'](Vw)
        Vn = Sm + np.einsum('nij,nj->ni', M, d['delta'])
        a = d['a'][:, None]
        Vw2 = Vw * (1 - a) + Vn * a
        F = d['F']
        fn = np.cross(Vw2[F[:, 1]] - Vw2[F[:, 0]], Vw2[F[:, 2]] - Vw2[F[:, 0]])
        N = np.zeros_like(Vw2); np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        return Vw2[d['inv']], (N[d['inv']], d['a'][d['inv']])

    def pose_mats(s, G, Pm, b, apply=True):
        from .rig import _M
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        s.G, s.Wp = G, Pm
        KQ, KD = _dq_de_matrices(K)
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        ropa = bool(getattr(s, 'ropa', None)) and apply
        Vs, Ns = [], []
        for k, p in enumerate(s.prims):
            R, t = s._mezcla(k, K, KQ, KD)
            V = np.einsum('vij,vj->vi', R, p.get('Pm', p['P'])) + t
            Nn = np.einsum('vij,vj->vi', R, p['N'])
            if s.ALISAR_CODO and p['mname'].startswith('Std_Skin_Arm'):
                V, extra = s._alisar(k, V)
                if extra is not None:
                    Nz, aa = extra
                    Nn = Nn * (1 - aa[:, None]) + Nz * aa[:, None]
            V = V @ Cz.T; Nn = Nn @ Cz.T
            if apply: V = (V - np.array([0, 0, s.base.zoff])) * s.base.scale
            Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            Vs.append(V); Ns.append(Nn)
        if ropa:
            s._fisica(Vs, Ns)
        return [(V[p['idx']], Nn[p['idx']]) for V, Nn, p in zip(Vs, Ns, s.prims)]


# ------------------------------------------------------------------------------------------------ dedos naturales
class _DedosNaturales:
    """Mano NATURAL (soma/manos.py) cuando la animación no mueve los dedos: Kimodo deja los dedos rectos, pegados unos
    a otros y todos con la misma curva (se ven torcidos). Aquí se abren en abanico, se cierran más de índice a meñique y
    el pulgar queda suelto. Si un agarre mueve los dedos, se respetan: la mano natural se va apagando sola a medida
    que los dedos se apartan de la pose de Kimodo (sin saltos al entrar o salir del agarre)."""
    DEDOS_NATURALES = True
    TOLERANCIA_DEDOS = 0.25            # rad: más que esto de diferencia con la pose de Kimodo = la animación los mueve

    def _tabla_dedos(s):
        if not hasattr(s, '_tdedos'):
            import os
            d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_v7_mano_natural.npz'))
            from scipy.spatial.transform import Rotation as Ro
            t = {}
            for n, D, N in zip(d['nombres'], d['defecto'], d['natural']):
                n = str(n)
                if n in s.idx:
                    t[n] = (D, Ro.from_matrix(N @ D.T).as_rotvec())
            orden = sorted(t, key=lambda n: s.idx[n])
            s._tdedos = (t, orden)
        return s._tdedos

    def _dedos(s, G, Pm, b):
        from scipy.spatial.transform import Rotation as Ro
        t, orden = s._tabla_dedos()
        hechos = set()
        s._dedos_libres = {'Left': 1.0, 'Right': 1.0}
        for n in orden:
            j = s.idx[n]; p = s.padres[j]
            D, rv = t[n]
            Gp_old = s._Gviejo.get(p, G[p])
            Lc = Gp_old.T @ G[j]                              # giro local que trae la animación
            dev = np.linalg.norm(Ro.from_matrix(Lc @ D.T).as_rotvec())
            w = float(np.clip(1 - dev / s.TOLERANCIA_DEDOS, 0, 1))
            lado = 'Left' if n.startswith('Left') else 'Right'
            s._dedos_libres[lado] = min(s._dedos_libres[lado], w)
            s._Gviejo[j] = G[j].copy()
            G[j] = G[p] @ (Ro.from_rotvec(rv * w).as_matrix() @ Lc)
            hechos.add(j)
        # posiciones de todos los dedos (y las puntas "End") con los giros nuevos
        for L in ('Left', 'Right'):
            raiz = s.idx[L + 'Hand']
            for j in range(len(s.nombres)):
                if j in hechos or s.nombres[j].startswith(L + 'Hand') and j != raiz:
                    p = s.padres[j]
                    if s.nombres[j].endswith('End') and j not in hechos:
                        Lc = s._Gviejo.get(p, G[p]).T @ G[j]
                        G[j] = G[p] @ Lc
                    Pm[j] = Pm[p] + G[p] @ b['off'][j]

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if s.DEDOS_NATURALES:
            s._Gviejo = {}
            s._dedos(G, Pm, b)
        return G, Pm, b


# ------------------------------------------------------------------------------------------------ brazos sin temblor
class _SinTemblor:
    """Kimodo deja un temblor pequeño en los brazos (la mano libre "tiembla" o "patina" aunque no haga nada).
    Se alisa el movimiento de clavícula, brazo, antebrazo y mano (promedio de unos pocos cuadros vecinos) SOLO
    mientras esa mano está libre: si la mano agarra algo (los dedos los mueve el agarre) no se toca, y el alisado
    se apaga suave unos cuadros antes y después del agarre."""
    SIN_TEMBLOR = True
    TEMBLOR_SIGMA = 1.6           # cuadros (a 30 fps)

    def _clip_suave(s, clip):
        if not s.SIN_TEMBLOR:
            return clip
        c = s.__dict__.setdefault('_clips_suaves', {})
        if id(clip) in c:
            return c[id(clip)][1]
        import copy
        from scipy.spatial.transform import Rotation as Ro
        import os
        tabla = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_v7_mano_natural.npz'))
        nuevo = copy.copy(clip)
        L = clip.L.copy(); n = clip.n
        k = np.arange(-4, 5); wk = np.exp(-0.5 * (k / s.TEMBLOR_SIGMA) ** 2); wk /= wk.sum()
        for lado in ('Left', 'Right'):
            # mano libre por cuadro: dedos en la pose de Kimodo
            dev = np.zeros(n)
            for nom, D in zip(tabla['nombres'], tabla['defecto']):
                nom = str(nom)
                if not nom.startswith(lado) or nom not in clip.idx:
                    continue
                j = clip.idx[nom]
                R = np.einsum('nji,jk->nik', clip.L[:, j], D)          # L^T·D
                dev = np.maximum(dev, Ro.from_matrix(R).magnitude())
            libre = np.clip(1 - dev / 0.25, 0, 1)
            for _ in range(10):                                     # se apaga antes y después del agarre
                libre = np.minimum(libre, np.minimum(np.r_[libre[0], libre[:-1]], np.r_[libre[1:], libre[-1]]))
            if libre.max() <= 0:
                continue
            for nom in ('Shoulder', 'Arm', 'ForeArm', 'Hand'):
                j = clip.idx.get(lado + nom)
                if j is None:
                    continue
                q = Ro.from_matrix(clip.L[:, j]).as_quat()
                for f in range(1, n):
                    if q[f] @ q[f - 1] < 0: q[f] = -q[f]
                qs = np.zeros_like(q)
                for kk, w in zip(k, wk):
                    qs += w * q[np.clip(np.arange(n) + kk, 0, n - 1)]
                qs /= np.linalg.norm(qs, axis=1, keepdims=True)
                qf = q * (1 - libre[:, None]) + qs * libre[:, None]
                qf /= np.linalg.norm(qf, axis=1, keepdims=True)
                L[:, j] = Ro.from_quat(qf).as_matrix()
        G = np.zeros_like(clip.G); P = np.zeros_like(clip.P)
        for j in range(len(clip.nombres)):
            pa = clip.padres[j]
            if pa < 0:
                G[:, j] = L[:, j]; P[:, j] = clip.T[:, j]
            else:
                G[:, j] = G[:, pa] @ L[:, j]; P[:, j] = P[:, pa] + np.einsum('nij,nj->ni', G[:, pa], clip.T[:, j])
        nuevo.L, nuevo.G, nuevo.P = L, G, P
        if hasattr(nuevo, '_contactos'):
            nuevo._contactos = None
        c[id(clip)] = (clip, nuevo)
        return nuevo

    def globales(s, clip, t, bucle=False, pies=True):
        return super().globales(s._clip_suave(clip), t, bucle, pies)


class MaxV7(_SinTemblor, _PielSinQuebrarse, _ManoNoAtraviesaCara, _DedosNaturales, _PiernaIzquierda, MaxV6):
    """MaxV6 con la pierna izquierda arreglada."""


class MaxCaraV7(_SinTemblor, _PielSinQuebrarse, _PesoDelObjeto, _ManoNoAtraviesaCara, _DedosNaturales, _PiernaIzquierda,
                MaxCaraV6):
    """MaxCaraV6 (cara viva, hombros parejos, brazo derecho) con la pierna izquierda arreglada."""


# caras de los gestos nuevos del ZIP 29 (g21 mano en la cara, g22 brazos cruzados, g23 celebrar parejo)
from . import cara as _cara
_cara.GESTO_CARA.setdefault("celebrar_parejo", ("euforia", False))
_cara.GESTO_CARA.setdefault("brazos_cruzados", ("impaciente", False))
