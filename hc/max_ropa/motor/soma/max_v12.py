"""Max v11 (ZIP_33, archivo NUEVO: no cambia nada de lo que ya hay). Es MaxCaraV10 (ZIP 32) con los brazos arreglados
según la revisión del video de prueba:

A. BRAZO QUE CUELGA SUELTO, MÁS ESTIRADO: Kimodo deja el codo muy doblado cuando el brazo cuelga sin hacer nada (se
   veía "torcido"). Ahora, si el brazo cuelga y la mano está libre, se quita algo más de la mitad del doblez.
B. BRAZO ABIERTO SIN "PISTOLERO": para separar el brazo del cuerpo se abre desde el hombro, pero el antebrazo sigue
   colgando (antes se abría todo el brazo rígido y la mano se iba hacia afuera, como torcida).
C. CODO SIEMPRE HACIA AFUERA Y ABAJO: si el codo apunta hacia el pecho o se retuerce hacia atrás, se gira (alrededor del
   eje hombro-muñeca, la mano no se mueve) hasta que apunte afuera/abajo. El brazo que agarra ya no se retuerce para
   dejar aire de más: basta con 1 cm.
D. LA MANO NO SE METE EN LA CADERA NI EN EL PANTALÓN (bolsillos, muslos): antes solo se revisaba la camiseta.

Y todo lo del ZIP 32 (brazos planeados, pies clavados, respiración, camiseta). Uso:
    from soma.max_v11 import MaxCaraV12, MaxV12, PistaCaraV12

Lo del ZIP 32 que sigue igual: brazos planeados sobre la animación completa (los cuadros se pueden pedir en cualquier
orden), pies clavados sin patinar, respiración, camiseta (piel y pantalón no asoman, bordes lisos, pliegues).
"""
import numpy as np

from .max_v9 import (MaxV9, MaxCaraV9, PistaCaraV9, _ManoNoAtraviesaCara, _SinTemblor, _PielSinQuebrarse,
                     _PesoDelObjeto, _DedosNaturales, _PiernaIzquierda)
from .max_v6 import MaxV6, MaxCaraV6


PistaCaraV12 = PistaCaraV9


# ------------------------------------------------------------------------------------------------ utilidades del plan
def _suave01(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def _dilatar(x, paso):
    """y >= x, con cambio máximo 'paso' por cuadro (rampa hacia arriba ANTES y hacia abajo DESPUÉS de cada pico)."""
    y = np.asarray(x, float).copy()
    for f in range(1, len(y)):
        y[f] = max(y[f], y[f - 1] - paso)
    for f in range(len(y) - 2, -1, -1):
        y[f] = max(y[f], y[f + 1] - paso)
    return y


def _suavizar(y, sigma):
    if sigma <= 0 or len(y) < 3:
        return y
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2); w /= w.sum()
    yp = np.pad(y, (len(k) // 2, len(k) // 2), mode='edge')
    return np.convolve(yp, w, mode='valid')


def _plan_signo(x, paso, sigma):
    """Plan de una corrección con signo: cubre siempre lo que hace falta, con rampas suaves antes y después."""
    x = np.asarray(x, float)
    p = _dilatar(np.maximum(x, 0), paso); n = _dilatar(np.maximum(-x, 0), paso)
    y = _suavizar(p - n, sigma)
    y = np.where(x > 0, np.maximum(y, x), np.where(x < 0, np.minimum(y, x), y))
    return y


class _BrazosPlaneados:
    """Brazos afuera, codo del que agarra y mano libre fuera de cara/cuello/cuerpo, PLANEADOS sobre toda la animación."""
    PLAN_BRAZOS = True
    CODO_AFUERA = True
    HOLGURA_BRAZO = 0.035          # m de aire de verdad entre el brazo y la camiseta (codos siempre afuera)
    ABRIR_MAX = 32.0
    PLAN_SIGMA = 2.5               # cuadros de suavizado del plan
    # rapidez máxima de cada corrección (por segundo)
    VEL_ABRIR = 90.0
    VEL_GIRO_CODO = 180.0
    VEL_EMPUJE = 0.30
    _RAD_HOLG = np.array([0.040, 0.045, 0.042, 0.035, 0.032])

    # ---- distancia de verdad a la camiseta: si el punto no está "debajo" de la superficie (el vector al vértice más
    # cercano no va en la dirección de la normal), está afuera, a la distancia del vértice (bordes, cuello, axila)
    @staticmethod
    def _aire(sup, pts):
        arbol, V, Nn = sup
        d, j = arbol.query(pts)
        sd = np.einsum('ij,ij->i', pts - V[j], Nn[j])
        return np.where(sd < -0.7 * d, sd, d), j

    HOLGURA_CODO = 0.010           # al girar el codo (brazo que agarra) basta con 1 cm: girar más lo retuerce

    def _falta_holgura(s, sup, S, E, H, holgura=None):
        pts = np.array([E, S + (E - S) * 0.55, S + (E - S) * 0.8, E + (H - E) * 0.35, E + (H - E) * 0.7])
        cl, _ = s._aire(sup['tronco'], pts)
        return float((s._RAD_HOLG + (s.HOLGURA_BRAZO if holgura is None else holgura) - cl).max())

    # el brazo se abre desde el hombro, pero el antebrazo NO se abre igual (si no, la mano se va hacia afuera y el
    # brazo queda "de pistolero"): el codo sale y el antebrazo sigue colgando casi como venía
    ANTEBRAZO_ACOMPANA = 0.35       # 0 = el antebrazo no gira nada con la apertura, 1 = gira todo (como antes)

    def _abierto(s, eje, S, E, H, ab):
        R = s._rot_eje(eje, np.radians(ab)); Rc = s._rot_eje(eje, np.radians(ab * s.ANTEBRAZO_ACOMPANA))
        E2 = S + R @ (E - S)
        return E2, E2 + Rc @ (H - E)

    # ---- la mano y el antebrazo tampoco se meten en el PANTALÓN (cadera, bolsillos, muslos)
    MANO_PANTALON = True

    def _pantalon_region(s):
        if not hasattr(s, '_preg'):
            ks = [k for k, p in enumerate(s.prims) if p['mname'].startswith('Pants')]
            s._preg = None
            if ks:
                k = ks[0]; p = s.prims[k]
                v = np.unique(p.get('idx_orig', p['idx']))
                s._preg = (p['P'][v], p['N'][v], s.Jn[k][v], s.Wn[k][v])
        return s._preg

    def _superficie_cuerpo(s, G, Pm, b):
        out = super()._superficie_cuerpo(G, Pm, b)
        reg = s._pantalon_region() if s.MANO_PANTALON else None
        if reg is not None:
            from scipy.spatial import cKDTree
            from .rig import _M
            P, N, J, W = reg
            K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
            M = np.einsum('vk,vkij->vij', W, K[J])
            V = np.einsum('vij,vj->vi', M[:, :3, :3], P) + M[:, :3, 3]
            Nn = np.einsum('vij,vj->vi', M[:, :3, :3], N); Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            out['pant'] = (cKDTree(V), V, Nn)
        return out

    def _hundimiento_cuerpo(s, G, Pm, b, L, sup):
        h, n = super()._hundimiento_cuerpo(G, Pm, b, L, sup)
        if 'pant' not in sup:
            return h, n
        h2, n2 = super()._hundimiento_cuerpo(G, Pm, b, L, {L: sup['pant']})
        return (h2, n2) if h2 > h else (h, n)

    def _abrir_necesario(s, G, Pm, sup, L):
        ix = s.idx
        S, E, H = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']], Pm[ix[L + 'Hand']]
        if s._falta_holgura(sup, S, E, H) <= 0:
            return 0.0
        eje = s._eje_abrir(G, Pm, L)
        for a in np.arange(1.0, s.ABRIR_MAX + 0.01, 1.0):
            E2, H2 = s._abierto(eje, S, E, H, a)
            if s._falta_holgura(sup, S, E2, H2) <= 0:
                return float(a)
        return float(s.ABRIR_MAX)

    # el codo apunta hacia AFUERA y abajo (nunca hacia el pecho ni retorcido hacia atrás)
    CODO_GIRO_ORIENTA = 40.0        # como máximo se gira esto para orientar el codo (más, retuerce el brazo)
    CODO_DESVIO_MAX = 35.0          # grados que se deja apartar el codo de "afuera y abajo"

    def _codo_preferido(s, Pm, L):
        ix = s.idx
        otro = 'Right' if L == 'Left' else 'Left'
        afuera = Pm[ix[L + 'Arm']] - Pm[ix[otro + 'Arm']]; afuera[1] = 0
        afuera /= np.linalg.norm(afuera) + 1e-9
        p = afuera * 0.45 + np.array([0, -0.9, 0])
        return p / np.linalg.norm(p)

    def _codo_orientacion(s, Pm, L):
        """Giro mínimo (grados, alrededor del eje hombro-muñeca) para que el codo apunte afuera/abajo."""
        ix = s.idx
        S, E, H = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']], Pm[ix[L + 'Hand']]
        ax = H - S; ax /= np.linalg.norm(ax) + 1e-9
        d = (E - S) - ax * ((E - S) @ ax)
        q = s._codo_preferido(Pm, L); q = q - ax * (q @ ax)
        if np.linalg.norm(d) < 0.01 or np.linalg.norm(q) < 0.45:     # brazo estirado o hacia afuera/abajo: nada que girar
            return 0.0
        d /= np.linalg.norm(d); q /= np.linalg.norm(q)
        ang = np.degrees(np.arctan2(np.cross(d, q) @ ax, d @ q))      # de d a q, alrededor de ax
        if abs(ang) <= s.CODO_DESVIO_MAX:
            return 0.0
        return float(np.sign(ang) * min(abs(ang) - s.CODO_DESVIO_MAX, s.CODO_GIRO_ORIENTA))

    def _codo_necesario(s, G, Pm, sup, L, previo):
        ix = s.idx
        S, E, H = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']], Pm[ix[L + 'Hand']]
        a0 = s._codo_orientacion(Pm, L) if s.CODO_AFUERA else 0.0
        f = lambda E2: max(s._pen_brazo(sup, L, S, E2, H), s._falta_holgura(sup, S, E2, H, s.HOLGURA_CODO))
        E0 = S + s._rot_eje(H - S, np.radians(a0)) @ (E - S)
        if f(E0) <= 0.0:
            return a0
        mejor = None
        cand = np.arange(-s.GIRO_CODO_MAX, s.GIRO_CODO_MAX + 0.1, 4.0) + a0
        for a in sorted(cand, key=lambda a: abs(a - a0) + 0.3 * abs(a - previo)):
            R = s._rot_eje(H - S, np.radians(a))
            p = f(S + R @ (E - S))
            if p <= 0.0:
                return float(a)
            if mejor is None or p < mejor[0] - 1e-4:
                mejor = (p, float(a))
        return mejor[1]

    def _sub_de(s, raiz):
        c = s.__dict__.setdefault('_sub_raiz', {})
        if raiz not in c:
            dd = [raiz]
            for jj in range(len(s.nombres)):
                if jj not in dd and s.padres[jj] in dd: dd.append(jj)
            c[raiz] = dd
        return c[raiz]

    def _sub(s, L):
        c = s.__dict__.setdefault('_sub_brazo', {})
        if L not in c:
            iA = s.idx[L + 'Arm']; dd = [iA]
            for jj in range(len(s.nombres)):
                if jj not in dd and s.padres[jj] in dd: dd.append(jj)
            c[L] = dd
        return c[L]

    def _aplicar_abrir(s, G, Pm, L, ab):
        if ab <= 0.01:
            return
        eje = s._eje_abrir(G, Pm, L)
        S = Pm[s.idx[L + 'Arm']].copy(); R = s._rot_eje(eje, np.radians(ab))
        for jj in s._sub(L):
            G[jj] = R @ G[jj]; Pm[jj] = S + R @ (Pm[jj] - S)
        # el antebrazo vuelve casi a como colgaba (gira solo una parte de la apertura)
        Rc = s._rot_eje(eje, np.radians(-ab * (1 - s.ANTEBRAZO_ACOMPANA)))
        iF = s.idx[L + 'ForeArm']; E = Pm[iF].copy()
        for jj in s._sub_de(iF):
            G[jj] = Rc @ G[jj]; Pm[jj] = E + Rc @ (Pm[jj] - E)

    def _base(s, clip, t, bucle, pies):
        G, Pm, b = super(_ManoNoAtraviesaCara, s).globales(clip, t, bucle, pies)
        s._ultimo_b = b
        if s.BRAZO_RELAJADO:
            s._relajar_codos(G, Pm)
        if s.BALANCEO_BRAZOS:
            s._balancear(clip, t, G, Pm)
        return G, Pm, b

    # ---- al caminar, el brazo libre se balancea al revés de la pierna de su lado (Kimodo los deja casi quietos)
    BALANCEO_BRAZOS = True
    BALANCEO_GRADOS = 11.0          # lo que se SUMA al balanceo de Kimodo con el pie a 40 cm adelante/atrás

    def _caminando(s, clip):
        c = s.__dict__.setdefault('_camina', {})
        if id(clip) not in c:
            from scipy.ndimage import gaussian_filter1d
            P = clip.P[:, clip.idx['Hips']][:, [0, 2]]
            v = np.r_[0, np.linalg.norm(np.diff(gaussian_filter1d(P, 2.0, axis=0), axis=0), axis=1)] * clip.fps
            w = _suave01((v - 0.35) / 0.3)
            c[id(clip)] = (clip, gaussian_filter1d(w, 0.15 * clip.fps))
        return c[id(clip)][1]

    def _balancear(s, clip, t, G, Pm):
        w = s._caminando(clip)
        f = min(max(t * clip.fps, 0.0), clip.n - 1.0); i = int(np.floor(f)); j = min(i + 1, clip.n - 1)
        wk = float(w[i] * (1 - (f - i)) + w[j] * (f - i))
        if wk <= 1e-3:
            return
        ix = s.idx
        fwd = G[ix['Chest']] @ s._fwd_pecho(); fwd[1] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
        libres = getattr(s, '_dedos_libres', {'Left': 1.0, 'Right': 1.0})
        for L in ('Left', 'Right'):
            lw = _suave01((libres.get(L, 1.0) - 0.7) / 0.25)
            if lw <= 0:
                continue
            adelante = (Pm[ix[L + 'Foot']] - Pm[ix['Hips']]) @ fwd          # su pie adelante -> el brazo atrás
            ang = -np.clip(adelante / 0.4, -1, 1) * s.BALANCEO_GRADOS * wk * lw
            iA = ix[L + 'Arm']; S = Pm[iA].copy()
            eje = np.cross(np.array([0, 1.0, 0]), fwd)                      # eje de lado a lado
            R = s._rot_eje(eje, np.radians(ang))
            if ang * ((R @ (Pm[ix[L + 'Hand']] - S)) @ fwd - (Pm[ix[L + 'Hand']] - S) @ fwd) < 0:
                R = s._rot_eje(eje, -np.radians(ang))
            for jj in s._sub(L):
                G[jj] = R @ G[jj]; Pm[jj] = S + R @ (Pm[jj] - S)

    # ---- brazo que cuelga suelto: Kimodo lo deja muy doblado en el codo (se ve "torcido"); se estira un poco
    BRAZO_RELAJADO = True
    RELAJA = 0.55                   # cuánto del doblez (por encima de 8°) se quita cuando el brazo cuelga suelto

    def _relajar_codos(s, G, Pm):
        ix = s.idx
        libres = getattr(s, '_dedos_libres', {'Left': 1.0, 'Right': 1.0})
        for L in ('Left', 'Right'):
            lw = _suave01((libres.get(L, 1.0) - 0.7) / 0.25)        # solo la mano del todo libre
            if lw <= 0:
                continue
            S, E, H = Pm[ix[L + 'Arm']], Pm[ix[L + 'ForeArm']], Pm[ix[L + 'Hand']]
            u, f = E - S, H - E
            nu, nf = np.linalg.norm(u), np.linalg.norm(f)
            cuelga = np.degrees(np.arccos(np.clip(-u[1] / nu, -1, 1)))        # 0 = brazo hacia abajo
            doblez = np.degrees(np.arccos(np.clip(u @ f / (nu * nf), -1, 1)))
            w = lw * _suave01((50.0 - cuelga) / 15.0) * _suave01((75.0 - doblez) / 20.0)
            quita = (doblez - 8.0) * s.RELAJA * w
            if quita <= 0.2:
                continue
            eje = np.cross(u, f)
            if np.linalg.norm(eje) < 1e-6:
                continue
            R = s._rot_eje(-eje, np.radians(quita))
            iF = ix[L + 'ForeArm']
            for jj in s._sub_de(iF):
                if jj == iF:
                    G[jj] = R @ G[jj]
                else:
                    G[jj] = R @ G[jj]; Pm[jj] = E + R @ (Pm[jj] - E)

    def _plan(s, clip):
        c = s.__dict__.setdefault('_planes', {})
        if id(clip) in c:
            return c[id(clip)][1]
        n = clip.n; fps = clip.fps
        bases, libres = [], []
        s._planeando = True
        for f in range(n):
            G, Pm, b = s._base(clip, f / fps, False, True)
            bases.append((G.copy(), Pm.copy()))
            libres.append(dict(getattr(s, '_dedos_libres', {'Left': 1.0, 'Right': 1.0})))
        s._ultimo_b = b
        LADOS = ('Left', 'Right')
        # A) abrir el brazo libre
        nec = {L: np.zeros(n) for L in LADOS}
        for f, (G, Pm) in enumerate(bases):
            sup = s._superficie_cuerpo(G, Pm, b)
            for L in LADOS:
                if libres[f][L] >= 0.5 and s.MANO_CUERPO:
                    nec[L][f] = s._abrir_necesario(G, Pm, sup, L)
        ab = {L: np.maximum(_suavizar(_dilatar(nec[L], s.VEL_ABRIR / fps), s.PLAN_SIGMA), nec[L]) for L in LADOS}
        for L in LADOS:            # el brazo que agarra no se abre (la mano se queda en el agarre)
            ab[L] = ab[L] * _suave01((np.array([lb[L] for lb in libres]) - 0.3) / 0.4)
        # B) girar el codo (también el que agarra)
        necc = {L: np.zeros(n) for L in LADOS}
        prev = {L: 0.0 for L in LADOS}
        for f, (G0, P0) in enumerate(bases):
            G, Pm = G0.copy(), P0.copy()
            for L in LADOS:
                s._aplicar_abrir(G, Pm, L, ab[L][f])
            sup = s._superficie_cuerpo(G, Pm, b)
            for L in LADOS:
                if s.MANO_CUERPO:
                    necc[L][f] = prev[L] = s._codo_necesario(G, Pm, sup, L, prev[L])
        codo = {L: _plan_signo(necc[L], s.VEL_GIRO_CODO / fps, s.PLAN_SIGMA) for L in LADOS}
        # C) mano libre fuera de cara, cuello y cuerpo
        nece = {L: np.zeros((n, 3)) for L in LADOS}
        cara_mm = np.zeros(n); cuerpo_mm = np.zeros(n)
        ih = s.idx['Head']
        for f, (G0, P0) in enumerate(bases):
            G, Pm = G0.copy(), P0.copy()
            for L in LADOS:
                s._aplicar_abrir(G, Pm, L, ab[L][f]); s._aplicar_giro_codo(G, Pm, b, L, codo[L][f])
            sup = s._superficie_cuerpo(G, Pm, b)
            s.mano_cara_mm = 0.0; s.mano_cuerpo_mm = 0.0
            for L in LADOS:
                if libres[f][L] >= 0.5:
                    sc = None
                    if s.MANO_CARA and np.linalg.norm(Pm[s.idx[L + 'Hand']] - Pm[ih]) <= 0.35:
                        sc = s._superficie(G, Pm, b)
                    nece[L][f] = s._empuje(G, Pm, b, L, sc, sup if s.MANO_CUERPO else None)
            cara_mm[f] = s.mano_cara_mm; cuerpo_mm[f] = s.mano_cuerpo_mm
        emp = {L: np.stack([_plan_signo(nece[L][:, k], s.VEL_EMPUJE / fps, s.PLAN_SIGMA) for k in range(3)], 1)
               for L in LADOS}
        # la mano que agarra no se mueve nunca: lo planeado para la mano libre se apaga suave al agarrar
        for L in LADOS:
            lw = _suave01((np.array([lb[L] for lb in libres]) - 0.3) / 0.4)
            emp[L] = emp[L] * lw[:, None]
        s._planeando = False
        plan = dict(ab=ab, codo=codo, emp=emp, cara_mm=cara_mm, cuerpo_mm=cuerpo_mm, n=n, fps=fps)
        c[id(clip)] = (clip, plan)
        return plan

    @staticmethod
    def _en(arr, f):
        f = min(max(f, 0.0), len(arr) - 1.0); i = int(np.floor(f)); j = min(i + 1, len(arr) - 1); w = f - i
        return arr[i] * (1 - w) + arr[j] * w

    def globales(s, clip, t, bucle=False, pies=True):
        if not s.PLAN_BRAZOS or bucle or not pies or getattr(s, '_planeando', False):
            return super().globales(clip, t, bucle, pies)
        plan = s._plan(clip)
        G, Pm, b = s._base(clip, t, bucle, pies)
        s._ultimo_b = b
        f = t * plan['fps']
        s.mano_cara_mm = float(s._en(plan['cara_mm'], f)); s.mano_cuerpo_mm = float(s._en(plan['cuerpo_mm'], f))
        for L in ('Left', 'Right'):
            s._aplicar_abrir(G, Pm, L, float(s._en(plan['ab'][L], f)))
            s._aplicar_giro_codo(G, Pm, b, L, float(s._en(plan['codo'][L], f)))
            d = s._en(plan['emp'][L], f)
            if np.linalg.norm(d) > 1e-4:
                s._ik_mano(G, Pm, b, L, Pm[s.idx[L + 'Hand']] + d)
        return G, Pm, b


# ------------------------------------------------------------------------------------------------ respiración
class _Respira:
    """Respiración: el pecho y el abdomen se expanden y vuelven, suave (piel y camiseta juntas). No se mueve ningún
    hueso: hombros, brazos, manos, agarres y cabeza quedan exactamente igual. Con esfuerzo (objeto pesado) respira
    más rápido y más hondo."""
    RESPIRA = True
    RESPIRA_MM = 6.0            # mm que sale el pecho (adelante) al tomar aire
    RESPIRA_SEG = 4.2           # segundos por respiración en reposo
    RESPIRA_ESFUERZO_SEG = 2.6  # con esfuerzo máximo

    def _mascara_respira(s):
        if not hasattr(s, '_mresp'):
            ix = s.idx
            tronco = [ix[n] for n in ('Spine1', 'Spine2', 'Chest') if n in ix]
            fuera = [ix[n] for n in s.nombres if n.endswith(('Shoulder', 'Arm', 'ForeArm', 'Neck1', 'Neck2', 'Head'))]
            cadera = [ix['Hips']] + [ix[n] for n in ('LeftLeg', 'RightLeg') if n in ix]
            s._mresp = {}
            for k, p in enumerate(s.prims):
                if not p['mname'].startswith(('Poloche', 'Std_Skin_Body')):
                    continue
                Jn, Wn = s.Jn[k], s.Wn[k]
                w = (Wn * np.isin(Jn, tronco)).sum(1)
                w = w * np.clip(1 - 2.5 * (Wn * np.isin(Jn, fuera)).sum(1), 0, 1)
                w = w * np.clip(1 - 2.0 * (Wn * np.isin(Jn, cadera)).sum(1), 0, 1)
                N = p['N']
                dirf = 0.20 + 0.80 * np.clip(N[:, 2], 0, 1) + 0.35 * np.abs(N[:, 0])
                m = w * dirf
                if m.max() > 1e-4:
                    s._mresp[k] = (m[:, None] * N).astype(np.float64)
        return s._mresp

    def _fase_respira(s, t):
        """0..1 (cuánto aire tiene), con el ritmo que cambia suave según el esfuerzo (sin saltos)."""
        carga = getattr(s, '_carga', None)
        esf = (lambda u: carga(u)[0]) if carga is not None else (lambda u: 0.0)
        per = lambda u: s.RESPIRA_SEG * (1 - esf(u)) + s.RESPIRA_ESFUERZO_SEG * esf(u)
        h = 0.05; n = int(max(t, 0.0) / h)
        fase = sum(h / per((i + 0.5) * h) for i in range(n)) + (max(t, 0.0) - n * h) / per(t)
        x = (fase + 0.3) % 1.0
        e = esf(t)
        # tomar aire 40 % del ciclo, soltar 60 %
        v = 0.5 - 0.5 * np.cos(np.pi * x / 0.4) if x < 0.4 else 0.5 + 0.5 * np.cos(np.pi * (x - 0.4) / 0.6)
        return float(v) * (1.0 + 0.6 * e)

    def pose_mats(s, G, Pm, b, apply=True):
        if not s.RESPIRA:
            return super().pose_mats(G, Pm, b, apply)
        a = s._fase_respira(float(getattr(s, '_t', 0.0) or 0.0)) * s.RESPIRA_MM / 1000.0
        mascara = s._mascara_respira()
        guardado = {}
        for k, D in mascara.items():
            p = s.prims[k]; guardado[k] = p.get('Pm')
            p['Pm'] = (guardado[k] if guardado[k] is not None else p['P']) + D * a
        try:
            return super().pose_mats(G, Pm, b, apply)
        finally:
            for k, v in guardado.items():
                if v is None:
                    del s.prims[k]['Pm']
                else:
                    s.prims[k]['Pm'] = v


# ------------------------------------------------------------------------------------------------ camiseta lisa
class _CamisetaLisa:
    """Camiseta: (a) la piel del brazo y del cuello y el pantalón nunca asoman a través de la tela (antes asomaban por
    el borde de las mangas y del cuello y, agachado con las piernas abiertas, el pantalón por el bajo); (b) los bordes
    quedan lisos; (c) pliegues suaves en la manga cuando la tela se encoge. Solo cambia la tela (no los huesos)."""
    BORDES_LISOS = True
    AXILA_LISA = False          # (probado: con la malla de esta camiseta abría una raja en la axila)
    PLIEGUES = True
    AXILA_RADIO = 0.15           # m alrededor del hombro
    PLIEGUE_MM = 2.5             # alto máximo del pliegue
    PLIEGUE_ONDA = 0.028         # m entre pliegues

    def _malla_tela(s, k):
        c = s.__dict__.setdefault('_mtela', {})
        if k in c:
            return c[k]
        from scipy.sparse import coo_matrix, diags
        from collections import defaultdict
        p = s.prims[k]
        key = np.round(p['P'] / 1e-5).astype(np.int64)
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
        n = len(rep)
        F = inv[p['idx_orig'] if 'idx_orig' in p else p['idx']].reshape(-1, 3)
        i = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
        j = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
        A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr(); A.data[:] = 1
        deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
        S = diags(1 / deg) @ A
        # bordes: aristas que tienen un solo triángulo, ordenadas en lazos
        E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
        u, cnt = np.unique(E, axis=0, return_counts=True); B = u[cnt == 1]
        ad = defaultdict(list)
        for a, b in B: ad[a].append(b); ad[b].append(a)
        vistos = set(); lazos = []
        for a0 in ad:
            if a0 in vistos: continue
            lz = [a0]; vistos.add(a0); prev = None; cur = a0
            while True:
                nx = [x for x in ad[cur] if x != prev and x not in vistos]
                if not nx: break
                prev, cur = cur, nx[0]; lz.append(cur); vistos.add(cur)
            if len(lz) >= 8:
                lazos.append(np.array(lz))
        # anillos cerca del borde (para repartir el alisado sin hacer escalón)
        borde = np.zeros(n, bool)
        for lz in lazos: borde[lz] = True
        r1 = (A @ borde.astype(float) > 0) & ~borde
        r2 = (A @ r1.astype(float) > 0) & ~borde & ~r1
        c[k] = dict(rep=rep, inv=inv, F=F, A=A, S=S, lazos=lazos, borde=borde, r1=r1, r2=r2)
        return c[k]

    def _bordes_lisos(s, k, V):
        d = s._malla_tela(k)
        if not d['lazos']:
            return V
        X = V[d['rep']].copy(); X0 = X.copy()
        for lz in d['lazos']:
            Y = X[lz]
            for _ in range(6):                        # Taubin: alisa sin encoger el borde
                for f in (0.5, -0.53):
                    Y = Y + f * ((np.roll(Y, 1, 0) + np.roll(Y, -1, 0)) / 2 - Y)
            X[lz] = Y
        D = X - X0
        # los anillos de al lado se mueven un poco (sin escalón)
        A = d['A']; cnt = np.asarray(A @ d['borde'].astype(float)).ravel()
        prom = (A @ (D * d['borde'][:, None])) / np.maximum(cnt, 1)[:, None]
        X[d['r1']] += 0.5 * prom[d['r1']]
        cnt2 = np.asarray(A @ d['r1'].astype(float)).ravel()
        D1 = np.zeros_like(D); D1[d['r1']] = 0.5 * prom[d['r1']]
        prom2 = (A @ D1) / np.maximum(cnt2, 1)[:, None]
        X[d['r2']] += 0.4 * prom2[d['r2']]
        return X[d['inv']]

    def _prep_axila(s, k):
        c = s.__dict__.setdefault('_paxila', {})
        if k in c:
            return c[k]
        d = s._malla_tela(k); p = s.prims[k]
        P0 = p['P'].astype(float)[d['rep']]
        a = np.zeros(len(P0))
        for L in ('Left', 'Right'):
            dist = np.linalg.norm(P0 - s.pA[L + 'Arm'], axis=1)
            a = np.maximum(a, np.clip(1.6 - 1.6 * dist / s.AXILA_RADIO, 0, 1))
        a[d['borde']] = 0                                 # los bordes los maneja _bordes_lisos
        F = d['F']; S = d['S']
        vec = np.full(len(P0), -1); vec[F[:, 0]] = F[:, 1]; vec[F[:, 1]] = F[:, 2]; vec[F[:, 2]] = F[:, 0]
        def marco(X):
            Sm = X.copy()
            for _ in range(10): Sm = 0.5 * Sm + 0.5 * (S @ Sm)
            fn = np.cross(Sm[F[:, 1]] - Sm[F[:, 0]], Sm[F[:, 2]] - Sm[F[:, 0]])
            N = np.zeros_like(Sm); np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            T = Sm[vec] - Sm; T -= N * np.einsum('ij,ij->i', T, N)[:, None]
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            return Sm, np.stack([T, np.cross(N, T), N], 2)
        Sm0, M0 = marco(P0)
        delta = np.einsum('nji,nj->ni', M0, P0 - Sm0)
        # para los pliegues: cuánto de cada vértice es manga (y el eje del brazo en reposo)
        Jn, Wn = s.Jn[k], s.Wn[k]
        manga = {}
        for L in ('Left', 'Right'):
            w = (Wn * np.isin(Jn, [s.idx[L + 'Arm']])).sum(1)[d['rep']]
            manga[L] = np.clip((w - 0.35) / 0.4, 0, 1)
        area0 = np.linalg.norm(np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]]), axis=1)
        c[k] = dict(a=a, marco=marco, delta=delta, manga=manga, area0=area0, P0=P0)
        return c[k]

    def _axila_lisa(s, k, V):
        d = s._malla_tela(k); q = s._prep_axila(k)
        X = V[d['rep']]; s._wpliegue = None
        if s.AXILA_LISA and q['a'].any():
            Sm, M = q['marco'](X)
            Xn = Sm + np.einsum('nij,nj->ni', M, q['delta'])
            X = X * (1 - q['a'][:, None]) + Xn * q['a'][:, None]
        if s.PLIEGUES:
            X = s._pliegues(k, X, d, q)
        return X[d['inv']]

    def _pliegues(s, k, X, d, q):
        F = d['F']
        area = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
        r = area / np.maximum(q['area0'], 1e-12)
        enc = np.zeros(len(X)); cnt = np.zeros(len(X))
        for col in range(3):
            np.add.at(enc, F[:, col], np.clip(1 - r, 0, 0.5)); np.add.at(cnt, F[:, col], 1)
        enc = enc / np.maximum(cnt, 1)
        for _ in range(3):
            enc = 0.5 * enc + 0.5 * (d['S'] @ enc)
        fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
        N = np.zeros_like(X); np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        out = X.copy(); wp = np.zeros(len(X))
        for L in ('Left', 'Right'):
            m = q['manga'][L]
            if not m.any(): continue
            eje = s.pA[L + 'ForeArm'] - s.pA[L + 'Arm']; eje /= np.linalg.norm(eje)
            u = (q['P0'] - s.pA[L + 'Arm']) @ eje
            w = np.clip(enc / 0.25, 0, 1) * m * (1 - d['borde'])
            out += N * (s.PLIEGUE_MM / 1000 * w * 0.5 * (1 + np.sin(2 * np.pi * u / s.PLIEGUE_ONDA)))[:, None]
            wp = np.maximum(wp, w)
        s._wpliegue = wp
        return out

    def pose_mats(s, G, Pm, b, apply=True):
        from .rig import _M
        from .max_v9 import _dq_de_matrices
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        s.G, s.Wp = G, Pm
        KQ, KD = _dq_de_matrices(K)
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        ropa = bool(getattr(s, 'ropa', None)) and apply
        Vs, Ns = [], []; piel = []
        for k, p in enumerate(s.prims):
            R, t = s._mezcla(k, K, KQ, KD)
            V = np.einsum('vij,vj->vi', R, p.get('Pm', p['P'])) + t
            Nn = np.einsum('vij,vj->vi', R, p['N'])
            if s.ALISAR_CODO and p['mname'].startswith('Std_Skin_Arm'):
                V, extra = s._alisar(k, V)
                if extra is not None:
                    Nz, aa = extra
                    Nn = Nn * (1 - aa[:, None]) + Nz * aa[:, None]
            if p['mname'].startswith(('Std_Skin_Arm', 'Std_Skin_Body', 'Std_Skin_Head', 'Pants')):
                vis = s.__dict__.setdefault('_piel_vis', {})
                if k not in vis:                       # solo la piel que se dibuja (la de debajo de la tela no se ve)
                    vis[k] = np.unique(p['idx'])
                if len(vis[k]):
                    grupo = 'brazo' if p['mname'].startswith('Std_Skin_Arm') else \
                        ('pantalon' if p['mname'].startswith('Pants') else 'cuerpo')
                    piel.append((grupo, (V[vis[k]], Nn[vis[k]])))
            if p['mname'].startswith('Poloche'):
                s._wpliegue = None
                if s.AXILA_LISA or s.PLIEGUES:
                    V = s._axila_lisa(k, V)
                # las normales de la tela se quedan las de siempre (lisas); solo en los pliegues se usan las de la forma
                if s._wpliegue is not None and s._wpliegue.any():
                    d_ = s._malla_tela(k)
                    w_ = s._wpliegue[d_['inv']][:, None]
                    Nn = Nn * (1 - w_) + s._normales(k, V, Nn) * w_
                if s.BORDES_LISOS:
                    V = s._bordes_lisos(k, V)
                if s.PIEL_ADENTRO_TELA and piel:
                    V = s._tela_sobre_piel(k, V, piel, 0.0025 if 'Interior' in p['mname'] else 0.0045)
            V = V @ Cz.T; Nn = Nn @ Cz.T
            if apply: V = (V - np.array([0, 0, s.base.zoff])) * s.base.scale
            Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            Vs.append(V); Ns.append(Nn)
        if ropa:
            s._fisica(Vs, Ns)
        return [(V[p['idx']], Nn[p['idx']]) for V, Nn, p in zip(Vs, Ns, s.prims)]

    PIEL_ADENTRO_TELA = True      # la piel y el pantalón nunca asoman a través de la camiseta
    TELA_ALISA_EMPUJE = 8         # (ZIP 34: antes 2) el empuje de la tela se reparte más: el dobladillo no se ondula

    def _tela_sobre_piel(s, k, V, piel, margen):
        """La manga se revisa contra la piel del brazo; el resto de la camiseta contra la piel del cuerpo y el cuello."""
        from scipy.spatial import cKDTree
        d = s._malla_tela(k); q = s._prep_axila(k)
        X = V[d['rep']]
        manga = np.maximum(q['manga']['Left'], q['manga']['Right']) > 0.5
        falta = np.zeros(len(X)); Nj = np.zeros_like(X)
        # (grupo, vértices de tela, distancia máx., hasta cuánto adentro se considera que asoma)
        for grupo, sel, dmax, hondo in (('brazo', manga, 0.03, 0.025), ('cuerpo', ~manga, 0.03, 0.025),
                                        ('pantalon', ~manga, 0.10, 0.10)):
            pz = [v for n, v in piel if n == grupo]
            if not pz or not sel.any():
                continue
            P = np.vstack([a for a, _ in pz]); N = np.vstack([b for _, b in pz])
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
            dist, j = cKDTree(P).query(X[sel])
            sd = np.einsum('ij,ij->i', X[sel] - P[j], N[j])
            f = np.where((dist < dmax) & (sd > -hondo) & (margen - sd > 0), margen - sd, 0.0)
            idx = np.where(sel)[0]
            mas = f > falta[idx]
            falta[idx[mas]] = f[mas]; Nj[idx[mas]] = N[j[mas]]
        if not falta.any():
            return V
        D = Nj * falta[:, None]
        for _ in range(s.TELA_ALISA_EMPUJE):              # sin bultos ni ondas: el empuje se reparte con los vecinos
            D = 0.5 * D + 0.5 * (d['S'] @ D)
        n = np.einsum('ij,ij->i', D, Nj)
        D = D + Nj * np.clip(falta - n, 0, None)[:, None]   # nunca menos de lo que hace falta
        return (X + D)[d['inv']]

    def _normales(s, k, V, Nviejo):
        """Normales de la tela recalculadas de la forma nueva (mismo lado que las de antes)."""
        d = s._malla_tela(k); F = d['F']
        X = V[d['rep']]
        fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
        N = np.zeros_like(X); np.add.at(N, F[:, 0], fn); np.add.at(N, F[:, 1], fn); np.add.at(N, F[:, 2], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        N = N[d['inv']]
        sg = np.sign(np.einsum('ij,ij->i', N, Nviejo)); sg[sg == 0] = 1
        return N * sg[:, None]


# ------------------------------------------------------------------------------------------------ pies clavados
class _PiesClavados:
    """El pie apoyado no patina: mientras el talón está en el piso se clava el talón; cuando el talón se levanta
    (al dar el paso) se clava la punta. Solo se corrige el deslizamiento sobre el piso (la altura es la de la animación).
    Lo que hay que mover el pie se reparte SUAVE en el aire, entre un apoyo y el siguiente (sin saltos al soltar).

    PARARSE DERECHO (ZIP 34): cuando Max está de pie quieto (no camina ni se agacha), Kimodo lo deja con las rodillas
    dobladas y, a veces, con las piernas muy abiertas. Aquí: (a) los pies quedan a un ancho normal (se acercan lo que
    sobra; el cambio se hace en el aire, en el último paso, nunca arrastrando el pie) y (b) las rodillas se estiran
    subiendo la cadera (los pies no se mueven). La mano que agarra se queda en el objeto; la libre sube con el cuerpo."""
    PIES_CLAVADOS = True
    PIE_ALTO = 0.020          # m sobre el piso: más bajo que esto = apoyado
    PARARSE_DERECHO = True
    ANCHO_PIES = 0.32         # ancho normal entre tobillos (en las unidades del glb, como la cadera de Max)
    ESTIRA_RODILLA = 0.7      # cuánto del doblez de rodilla (por encima de 6°) se quita de pie
    RODILLA_AGACHADO = 40.0   # con la rodilla más doblada que esto es que se agacha a propósito: no se toca
    SUBE_MAX = 0.08           # lo más que sube la cadera

    def _crudo_pies(s, clip):
        c = s.__dict__.setdefault('_pies_crudo', {})
        if id(clip) not in c:
            ix = s.idx; n = clip.n
            LADOS = ('Left', 'Right')
            A = {L: np.zeros((n, 3)) for L in LADOS}; T = {L: np.zeros((n, 3)) for L in LADOS}
            Hp = {L: np.zeros((n, 3)) for L in LADOS}; K = {L: np.zeros((n, 3)) for L in LADOS}
            cad = np.zeros((n, 3))
            for f in range(n):
                G, Pm, b = super().globales(clip, f / clip.fps, False, False)
                cad[f] = Pm[ix['Hips']]
                for L in LADOS:
                    A[L][f] = Pm[ix[L + 'Foot']]; T[L][f] = Pm[ix[L + 'ToeBase']]
                    Hp[L][f] = Pm[ix[L + 'Leg']]; K[L][f] = Pm[ix[L + 'Shin']]
            c[id(clip)] = (clip, (A, T, Hp, K, cad))
        return c[id(clip)][1]

    def _de_pie(s, clip, A, cad, apoyos):
        """0..1 por cuadro: de pie quieto (los dos pies apoyados y la cadera casi sin avanzar)."""
        from scipy.ndimage import gaussian_filter1d, minimum_filter1d
        fps = clip.fps
        v = np.r_[0, np.linalg.norm(np.diff(gaussian_filter1d(cad[:, [0, 2]], 2.0, axis=0), axis=0), axis=1)] * fps
        w = _suave01((0.25 - v) / 0.12) * apoyos['Left'] * apoyos['Right']
        w = minimum_filter1d(w, size=int(0.4 * fps) | 1)
        return np.clip(gaussian_filter1d(w, 0.15 * fps), 0, 1)

    def _plan_pies(s, clip):
        c = s.__dict__.setdefault('_pies_plan', {})
        if id(clip) in c:
            return c[id(clip)][1]
        A, T, Hp, K, cad = s._crudo_pies(clip)
        n = clip.n; fps = clip.fps
        ya = np.percentile(np.r_[A['Left'][:, 1], A['Right'][:, 1]], 2)
        yt = np.percentile(np.r_[T['Left'][:, 1], T['Right'][:, 1]], 2)
        LADOS = ('Left', 'Right')
        apoyos, datos = {}, {}
        for L in LADOS:
            a, tp = A[L], T[L]
            ha, ht = a[:, 1] - ya, tp[:, 1] - yt
            apoyo = np.minimum(ha, ht) < s.PIE_ALTO
            for _ in range(2):
                apoyo = apoyo | (np.r_[apoyo[1:], False] & np.r_[False, apoyo[:-1]])
            apoyos[L] = apoyo.astype(float); datos[L] = (ha, ht, apoyo)
        quieto = s._de_pie(clip, A, cad, apoyos) if s.PARARSE_DERECHO else np.zeros(n)
        # (a) pies a un ancho normal: cuánto se acerca cada pie hacia el otro (por cuadro)
        acerca = {L: np.zeros((n, 3)) for L in LADOS}
        if s.PARARSE_DERECHO:
            for f in range(n):
                d = A['Left'][f] - A['Right'][f]; d[1] = 0
                ancho = np.linalg.norm(d)
                sobra = max(0.0, ancho - s.ANCHO_PIES)
                if sobra > 1e-4:
                    u = d / ancho
                    acerca['Left'][f] = -u * sobra / 2; acerca['Right'][f] = u * sobra / 2
        plan = {}
        for L in LADOS:
            a, tp = A[L], T[L]
            ha, ht, apoyo = datos[L]
            tramos = []; f = 0
            while f < n:
                if not apoyo[f]: f += 1; continue
                g = f
                while g + 1 < n and apoyo[g + 1]: g += 1
                if g - f >= 3:
                    tramos.append((f, g))
                f = g + 1
            off = np.zeros((n, 3)); tiene = np.zeros(n, bool)
            for f0, g0 in tramos:
                # el talón se clava en el punto medio de lo que patina (así la pierna nunca tiene que estirarse de más)
                plano = [q for q in range(f0, g0 + 1) if ha[q] - ht[q] < 0.010] or list(range(f0, g0 + 1))
                # el tramo entero se acerca igual (si en ese apoyo está de pie quieto): nunca arrastra el pie
                cuanto = _suave01((np.mean(quieto[f0:g0 + 1]) - 0.25) / 0.35)
                talon = np.median(a[plano], axis=0) + np.median(acerca[L][f0:g0 + 1], axis=0) * cuanto
                fp = min(plano, key=lambda q: np.linalg.norm((a[q] - talon)[[0, 2]]))
                punta = talon + (tp[fp] - a[fp])
                for q in range(f0, g0 + 1):
                    wt = _suave01((ha[q] - ht[q] - 0.010) / 0.025)          # talón levantado: clavar la punta
                    obj = talon * (1 - wt) + (punta - (tp[q] - a[q])) * wt
                    off[q] = obj - a[q]; off[q][1] = 0.0; tiene[q] = True        # solo en el piso (x, z)
            # entre apoyos: el desfase cambia suave en el aire
            idx = np.where(tiene)[0]
            if len(idx):
                for f in range(n):
                    if tiene[f]: continue
                    antes = idx[idx < f]; despues = idx[idx > f]
                    if len(antes) and len(despues):
                        fa, fb = antes[-1], despues[0]
                        w = _suave01((f - fa) / (fb - fa))
                        off[f] = off[fa] * (1 - w) + off[fb] * w
                    elif len(despues):
                        off[f] = off[despues[0]]
                    else:
                        off[f] = off[antes[-1]]
            plan[L] = off
        # (b) rodillas: cuánto sube la cadera para estirarlas (con los pies donde quedan clavados)
        sube = np.zeros(n)
        if s.PARARSE_DERECHO:
            for f in range(n):
                if quieto[f] <= 0.01:
                    continue
                dhs = []
                for L in LADOS:
                    H, Kn, An = Hp[L][f], K[L][f], A[L][f] + plan[L][f]
                    l1, l2 = np.linalg.norm(Kn - H), np.linalg.norm(A[L][f] - Kn)
                    u1, u2 = (Kn - H) / l1, (A[L][f] - Kn) / l2
                    flex = np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1)))
                    quita = (flex - 6.0) * s.ESTIRA_RODILLA * _suave01((s.RODILLA_AGACHADO - flex) / 10.0)
                    ft = np.radians(max(flex - max(quita, 0.0), 0.0))
                    dt = np.sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * np.cos(ft)) * 0.999
                    v = H - An; xz = v[0] ** 2 + v[2] ** 2
                    dhs.append(np.sqrt(max(dt * dt - xz, 0.0)) - v[1])
                sube[f] = np.clip(min(dhs), 0.0, s.SUBE_MAX) * quieto[f]
            from scipy.ndimage import gaussian_filter1d
            sube = gaussian_filter1d(sube, 0.12 * fps)
        plan['sube'] = sube; plan['quieto'] = quieto
        c[id(clip)] = (clip, plan)
        return plan

    def globales(s, clip, t, bucle=False, pies=True):
        if not s.PIES_CLAVADOS or bucle or not pies:          # en bucle: el bloqueo de antes
            return super().globales(clip, t, bucle, pies)
        G, Pm, b = super().globales(clip, t, bucle, False)
        plan = s._plan_pies(clip)
        f = min(max(t * clip.fps, 0.0), clip.n - 1.0); i = int(np.floor(f)); j = min(i + 1, clip.n - 1); w = f - i
        ix = s.idx
        objetivos = {}
        for L in ('Left', 'Right'):
            objetivos[L] = Pm[ix[L + 'Foot']] + plan[L][i] * (1 - w) + plan[L][j] * w
        dh = float(plan['sube'][i] * (1 - w) + plan['sube'][j] * w)
        if dh > 1e-4:
            manos = {L: Pm[ix[L + 'Hand']].copy() for L in ('Left', 'Right')}
            Pm += np.array([0.0, dh, 0.0])                     # todo el cuerpo sube...
            libres = getattr(s, '_dedos_libres', {'Left': 1.0, 'Right': 1.0})
            for L in ('Left', 'Right'):                          # ...menos la mano que agarra (se queda en el objeto)
                agarra = 1.0 - _suave01((libres.get(L, 1.0) - 0.48) / 0.3)
                if agarra > 1e-3:
                    s._ik_mano(G, Pm, b, L, Pm[ix[L + 'Hand']] * (1 - agarra) + manos[L] * agarra)
        for L in ('Left', 'Right'):
            if np.linalg.norm(objetivos[L] - Pm[ix[L + 'Foot']]) > 1e-5:
                s._ik_pierna(G, Pm, b, L, objetivos[L])
        return G, Pm, b


# ------------------------------------------------------------------------------------------------ plan guardado
class _PlanGuardado:
    """El plan de la animación (brazos, pies, postura) se guarda en disco la primera vez; si se vuelve a dibujar la
    MISMA animación con los mismos ajustes, el primer cuadro ya no tarda (se lee del disco en menos de 1 s).
    Carpeta: soma/planes_guardados/ (se puede borrar cuando se quiera: se vuelve a calcular sola)."""
    GUARDAR_PLAN = True
    CARPETA_PLANES = None            # None = soma/planes_guardados

    def _clave_plan(s, clip, que):
        import hashlib
        h = hashlib.md5()
        h.update(que.encode()); h.update(type(s).__name__.encode())
        h.update(np.ascontiguousarray(clip.L, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(clip.T, dtype=np.float32).tobytes())
        params = sorted((k, repr(getattr(s, k))) for k in dir(type(s))
                        if k.isupper() and isinstance(getattr(s, k), (int, float, bool, str, dict, tuple)))
        h.update(repr(params).encode())
        h.update(repr(getattr(getattr(s, 'pista', None), 'cargas', None)).encode())
        h.update(repr([(p['mname'], len(p['P'])) for p in s.prims]).encode())
        return h.hexdigest()[:20]

    def _carpeta_planes(s):
        import os
        d = s.CARPETA_PLANES or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'planes_guardados')
        os.makedirs(d, exist_ok=True)
        return d

    def _con_disco(s, que, clip, calcular):
        import os, pickle
        if not s.GUARDAR_PLAN:
            return calcular(clip)
        try:
            ruta = os.path.join(s._carpeta_planes(), '%s_%s.pkl' % (que, s._clave_plan(clip, que)))
        except Exception:
            return calcular(clip)
        if os.path.exists(ruta):
            try:
                with open(ruta, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        plan = calcular(clip)
        try:
            with open(ruta + '.tmp', 'wb') as f:
                pickle.dump(plan, f)
            os.replace(ruta + '.tmp', ruta)
        except Exception:
            pass
        return plan

    def _plan(s, clip):
        c = s.__dict__.setdefault('_planes', {})
        if id(clip) not in c:
            c[id(clip)] = (clip, s._con_disco('brazos', clip, lambda cl: super(_PlanGuardado, s)._plan(cl)))
        return c[id(clip)][1]

    def _plan_pies(s, clip):
        c = s.__dict__.setdefault('_pies_plan', {})
        if id(clip) not in c:
            c[id(clip)] = (clip, s._con_disco('pies', clip, lambda cl: super(_PlanGuardado, s)._plan_pies(cl)))
        return c[id(clip)][1]


class MaxV12(_PlanGuardado, _SinTemblor, _Respira, _CamisetaLisa, _PielSinQuebrarse, _BrazosPlaneados, _ManoNoAtraviesaCara, _PiesClavados, _DedosNaturales,
             _PiernaIzquierda, MaxV6):
    """MaxV9 + las mejoras del ZIP 32 (sin cara viva)."""


class MaxCaraV12(_PlanGuardado, _SinTemblor, _Respira, _CamisetaLisa, _PielSinQuebrarse, _PesoDelObjeto, _BrazosPlaneados, _ManoNoAtraviesaCara,
                 _PiesClavados, _DedosNaturales, _PiernaIzquierda, MaxCaraV6):
    """La que se usa. CABEZA (ZIP 34): al mirar abajo agacha menos el cuello y baja más los ojos, como una persona
    (antes: hasta 25° se dejaba igual y se levantaba como máximo 18°; ahora desde 15° y hasta 28°)."""
    CUELLO_LIBRE = 15.0
    CUELLO_QUITA = 0.75
    CUELLO_MAX_QUITA = 28.0
