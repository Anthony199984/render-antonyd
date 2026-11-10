"""ZIP 38 · CUERPO SIN ESQUELETO: Max se desploma "como una toalla mojada" cuando se le saca el esqueleto.

Desde el segundo `t0` el cuerpo de Max ya no lo sostienen los huesos: queda la piel con la carne (unos 64 kg de un
Max de 75 kg; el esqueleto, unos 11 kg, ya no está) y cae por su propio peso con física:
  - gravedad, choque y roce con el piso, y la mitad del impulso que traía (IMPULSO);
  - la carne casi no se estira ni se acorta a lo largo, pero se dobla y se aplasta con facilidad: las rodillas se
    doblan como goma y se desploma en el sitio, SIN REBOTE;
  - a los 0,7 s se asienta y se queda así (QUIETO_DESDE): pecho hundido y sin fuerza, cuello flojo, brazos tirados a
    los lados y la cabeza inclinada hacia atrás mirando arriba (CABEZA_ARRIBA = 35°);
  - el cuerpo no se atraviesa a sí mismo;
  - la CABEZA es rígida (no se deforma); la ropa, el pelo, los ojos y los dientes van pegados al cuerpo blando.
Uso:
    from soma.cuerpo_blando import CuerpoBlando
    cb = CuerpoBlando(m, clip, t0=6.0)          # m = MaxCaraV12 o V13 (con su pista); calcula la caída (2-3 min)
    res = cb.pose(t)                             # igual que m.pose(clip, t): antes de t0 es Max normal
"""
import numpy as np
from scipy.spatial import cKDTree

PIEL = ('Std_Skin_Head', 'Std_Skin_Body', 'Std_Skin_Arm', 'Std_Skin_Leg')


def _unicos(m, clip, t):
    """Posiciones de TODOS los vértices de cada pieza de Max en el tiempo t (también la piel que tapa la ropa).
    Devuelve (posiciones, usados): los vértices que no están en ningún triángulo quedan marcados como no usados."""
    guard = [p['idx'] for p in m.prims]
    for p in m.prims:
        if 'idx_sin_ropa' in p:
            p['idx'] = p['idx_sin_ropa']
    try:
        res = m.pose(clip, float(t))
    finally:
        for p, g in zip(m.prims, guard):
            p['idx'] = g
    out, usados = [], []
    for p, (V, _) in zip(m.prims, res):
        ix = p['idx_sin_ropa'] if 'idx_sin_ropa' in p else p['idx']
        U = np.zeros((len(p['P']), 3)); u = np.zeros(len(p['P']), bool)
        U[ix] = V; u[ix] = True
        out.append(U); usados.append(u)
    return out, usados


def _normales(V, F):
    n = np.zeros_like(V)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _polar(A):
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


class CuerpoBlando:
    PESO_TOTAL = 75.0          # kg de Max
    PESO_ESQUELETO = 11.0      # kg que se van con el esqueleto (~15 %)
    LADO_PARTICULA = 0.035     # m entre partículas del cuerpo blando
    FORMA = 0.03               # cuánto conserva cada pedacito de carne su forma (0 = líquido, 1 = rígido)
    PLASTICO = 0.03            # lo doblado se queda doblado (no vuelve a su forma: trapo mojado)
    SUBE_MAX = 0.15            # m/s: un trapo mojado no salta (sin rebote)
    IMPULSO = 0.5              # cuánto del impulso que traía conserva al quedarse sin esqueleto (1 = todo)
    QUIETO_DESDE = 0.7         # s después de quitar el esqueleto en que se asienta y se queda así (desplomado,
                               # cabeza arriba, brazos tirados a los lados); None = sigue cayendo libre
    ASIENTA_EN = 0.25          # s que tarda en quedarse quieto del todo
    CABEZA_ARRIBA = 35.0       # grados: la cabeza termina inclinada hacia atrás, mirando arriba (None = la deja caer)
    CUELLO_BLANDO = 0.15       # el cuello y la parte de arriba del pecho sin fuerza (1 = como el resto)
    TRONCO_BLANDO = 0.5        # el tronco se hunde más que brazos y piernas (1 = como el resto)
    CABEZA_DESDE = 0.3         # s después de quitar el esqueleto en que la cabeza empieza a irse hacia atrás (ya cayendo)
    CABEZA_TARDA = 0.5         # s que tarda en enderezarse
    ESTIRA = 0.8               # rigidez al estirarse (la carne casi no se estira)
    LARGO = 0.5                # a lo largo de brazos, piernas y tronco tampoco se acorta (mantiene su largo)
    ROCE = 0.85                # roce con el piso por paso (0 = hielo, 1 = pegado)
    AMORTIGUA = 1.5            # pérdida de velocidad por segundo (carne mojada: no rebota)
    SUBPASOS = 8
    ITERACIONES = 4
    DURACION = 2.5             # s que se simulan después de t0 (después queda quieto)
    FPS = 60

    def __init__(s, m, clip, t0, duracion=None):
        s.m, s.clip, s.t0 = m, clip, float(t0)
        if duracion: s.DURACION = duracion
        dt = 1.0 / s.FPS
        V0, us = _unicos(m, clip, t0); Vm, _ = _unicos(m, clip, t0 - dt)
        s.V0 = V0
        nombres = [p['mname'] for p in m.prims]
        es_piel = [any(n.startswith(k) for k in PIEL) for n in nombres]
        es_cabeza = [n.startswith('Std_Skin_Head') for n in nombres]
        # 1) partículas: la piel completa de Max (también la que tapa la ropa) reducida a una rejilla + el interior
        P = np.concatenate([V0[i][us[i]] for i in range(len(V0)) if es_piel[i]])
        Pv = np.concatenate([((V0[i] - Vm[i]) / dt)[us[i]] for i in range(len(V0)) if es_piel[i]])
        cab = np.concatenate([np.full(us[i].sum(), es_cabeza[i]) for i in range(len(V0)) if es_piel[i]])
        q = np.floor(P / s.LADO_PARTICULA).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True); inv = inv.ravel()
        n = inv.max() + 1
        cnt = np.bincount(inv, minlength=n).astype(float)
        X = np.stack([np.bincount(inv, P[:, k], n) for k in range(3)], 1) / cnt[:, None]
        Vel = np.stack([np.bincount(inv, Pv[:, k], n) for k in range(3)], 1) / cnt[:, None]
        cabeza = np.bincount(inv, cab.astype(float), n) / cnt > 0.5
        # interior: a lo largo de los huesos (sin la cabeza: esa es rígida)
        huesos = []
        for a, b in (('Hips', 'Spine1'), ('Spine1', 'Spine2'), ('Spine2', 'Chest'), ('Chest', 'Neck1'),
                     ('LeftLeg', 'LeftShin'), ('LeftShin', 'LeftFoot'), ('RightLeg', 'RightShin'), ('RightShin', 'RightFoot'),
                     ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'), ('RightArm', 'RightForeArm'),
                     ('RightForeArm', 'RightHand'), ('LeftShoulder', 'LeftArm'), ('RightShoulder', 'RightArm'),
                     ('Hips', 'LeftLeg'), ('Hips', 'RightLeg')):
            m.pose(clip, t0); A = m.joint_pos(a); B = m.joint_pos(b)
            m.pose(clip, t0 - dt); A1 = m.joint_pos(a); B1 = m.joint_pos(b)
            k = max(2, int(np.linalg.norm(B - A) / s.LADO_PARTICULA))
            for u in np.linspace(0, 1, k, endpoint=False):
                huesos.append((A + (B - A) * u, ((A - A1) + ((B - B1) - (A - A1)) * u) / dt,
                               (B - A) / (np.linalg.norm(B - A) + 1e-12)))
        if huesos:
            Hx = np.array([h[0] for h in huesos]); Hv = np.array([h[1] for h in huesos])
            # relleno: de cada punto de la piel (no cabeza) hacia su hueso más cercano (la carne de adentro)
            jh = cKDTree(Hx).query(X)[1]
            rel, relv = [], []
            for u in (0.3, 0.6, 0.85):
                rel.append(X[~cabeza] + (Hx[jh[~cabeza]] - X[~cabeza]) * u)
                relv.append(Vel[~cabeza] + (Hv[jh[~cabeza]] - Vel[~cabeza]) * u)
            Xi = np.r_[np.concatenate(rel), Hx]; Vi = np.r_[np.concatenate(relv), Hv]
            q = np.floor(Xi / s.LADO_PARTICULA + 0.5).astype(np.int64)
            _, inv2 = np.unique(q, axis=0, return_inverse=True); inv2 = inv2.ravel(); n2 = inv2.max() + 1
            c2 = np.bincount(inv2, minlength=n2).astype(float)
            Xi = np.stack([np.bincount(inv2, Xi[:, k], n2) for k in range(3)], 1) / c2[:, None]
            Vi = np.stack([np.bincount(inv2, Vi[:, k], n2) for k in range(3)], 1) / c2[:, None]
            lejos = cKDTree(X).query(Xi)[0] > 0.5 * s.LADO_PARTICULA          # no encima de la piel
            X = np.r_[X, Xi[lejos]]; Vel = np.r_[Vel, Vi[lejos]]
            cabeza = np.r_[cabeza, np.zeros(lejos.sum(), bool)]
        Vel = Vel * s.IMPULSO
        s.X0 = X.copy(); s.X = X.copy(); s.Vel = Vel.copy(); s.cabeza = cabeza
        N = len(X)
        # 2) peso: lo que queda sin el esqueleto, repartido según el grosor (las partículas de adentro pesan más)
        arbol = cKDTree(X)
        dens = np.array([len(v) for v in arbol.query_ball_point(X, 2.2 * s.LADO_PARTICULA)], float)
        masa = dens / dens.sum() * (s.PESO_TOTAL - s.PESO_ESQUELETO)
        s.masa = masa; s.w = 1.0 / masa
        # 3) uniones entre vecinas (la "carne"): largo de reposo
        pares = np.array(sorted(arbol.query_pairs(1.75 * s.LADO_PARTICULA)), int)
        pares = pares[~(cabeza[pares[:, 0]] & cabeza[pares[:, 1]])]           # dentro de la cabeza manda la cabeza rígida
        s.pares = pares
        s.largo = np.linalg.norm(X[pares[:, 0]] - X[pares[:, 1]], axis=1)
        # uniones a lo LARGO del brazo, la pierna o el tronco: tampoco se acortan (el cuerpo mantiene su largo, como
        # una toalla); las de través sí se aplastan (el cuerpo queda plano en el piso)
        if huesos:
            Hx_ = np.array([h[0] for h in huesos]); Hd = np.array([h[2] for h in huesos])
            eje = Hd[cKDTree(Hx_).query(X)[1]]
            e2 = eje[pares[:, 0]] + eje[pares[:, 1]]; e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12
            dd = X[pares[:, 1]] - X[pares[:, 0]]; dd /= np.linalg.norm(dd, axis=1, keepdims=True) + 1e-12
            s.a_lo_largo = np.abs((dd * e2).sum(1)) > 0.6
        else:
            s.a_lo_largo = np.zeros(len(pares), bool)
        s.vecinos = [[] for _ in range(N)]
        for a, b in pares:
            s.vecinos[a].append(b); s.vecinos[b].append(a)
        # pedacitos de carne: cada partícula con sus vecinas (conservan su forma, se doblan entre ellos)
        kmax = max(len(v) for v in s.vecinos) + 1
        C = np.full((N, kmax), -1, int)
        for i, v in enumerate(s.vecinos):
            C[i, 0] = i; C[i, 1:len(v) + 1] = v
        s.C = C; s.Cm = C >= 0; s.Cs = np.where(s.Cm, C, 0)
        s.ic = np.where(cabeza)[0]
        m.pose(clip, t0)
        up0 = m.joint_pos('HeadEnd') - m.joint_pos('Head'); up0 /= np.linalg.norm(up0)
        f0 = np.cross(m.joint_pos('LeftEye') - m.joint_pos('RightEye'), up0); f0 -= up0 * (f0 @ up0); f0 /= np.linalg.norm(f0)
        s.cab_marco0 = np.stack([f0, up0, np.cross(f0, up0)], 1)
        # dónde la carne no tiene fuerza: cuello y pecho alto (mucho), resto del tronco (algo)
        cuello = m.joint_pos('Neck1'); pecho = m.joint_pos('Chest'); cadera = m.joint_pos('Hips')
        def cerca_seg(A, B, r):
            ab = B - A; u = np.clip(((X - A) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(X - (A + u[:, None] * ab), axis=1) < r
        s.forma_p = np.ones(N)
        s.forma_p[cerca_seg(cadera, pecho, 0.20)] = s.TRONCO_BLANDO
        s.forma_p[cerca_seg(pecho, cuello, 0.13)] = s.CUELLO_BLANDO
        s.forma_p[cabeza] = 1.0
        s.Xc0 = X[s.ic] - X[s.ic].mean(0)
        s.radio = 0.45 * s.LADO_PARTICULA
        # 4) cada vértice de Max (piel, ropa, pelo, ojos...) pegado a sus partículas más cercanas
        s._atar()
        # 5) simular
        s._simular()

    def _atar(s):
        arbol = cKDTree(s.X0)
        s.atado = []
        for V in s.V0:
            d, j = arbol.query(V, k=6)
            w = 1.0 / (d + 0.01) ** 2; w /= w.sum(1, keepdims=True)
            s.atado.append((j, w))

    def _rotaciones(s, X):
        R = np.zeros((len(X), 3, 3))
        for i, vec in enumerate(s.vecinos):
            if len(vec) < 3:
                R[i] = np.eye(3); continue
            A = (X[vec] - X[i]).T @ (s.X0[vec] - s.X0[i])
            R[i] = _polar(A)
        if len(s.ic):
            c = X[s.ic].mean(0); Rc = _polar((X[s.ic] - c).T @ s.Xc0)
            R[s.ic] = Rc
        return R

    def _simular(s):
        dt = 1.0 / s.FPS / s.SUBPASOS
        g = np.array([0, 0, -9.81])
        a, b = s.pares[:, 0], s.pares[:, 1]
        wa, wb = s.w[a], s.w[b]
        Cs, Cm = s.Cs, s.Cm.astype(float)
        mc = s.masa[Cs] * Cm                                        # masas de cada pedacito
        Mc = mc.sum(1)
        X0c = s.X0[Cs]
        c0 = (X0c * mc[..., None]).sum(1) / Mc[:, None]
        Q0 = (X0c - c0[:, None]) * Cm[..., None]                    # forma de reposo de cada pedacito
        cuenta = np.bincount(Cs[s.Cm], minlength=len(s.X0)).astype(float)
        X, V = s.X.copy(), s.Vel.copy()
        frames = [X.copy()]
        nfr = int(s.DURACION * s.FPS)
        amort = np.exp(-s.AMORTIGUA * dt)
        vecino = set(map(tuple, np.sort(s.pares, 1)))
        for f in range(nfr):
            freno = 1.0
            if s.QUIETO_DESDE is not None and f / s.FPS > s.QUIETO_DESDE:
                u = min(1.0, (f / s.FPS - s.QUIETO_DESDE) / s.ASIENTA_EN)
                freno = 1.0 - 0.35 * u                                    # se asienta: pierde la velocidad
            for _ in range(s.SUBPASOS):
                V = (V + g * dt) * amort
                V[:, :2] *= freno ** (1.0 / s.SUBPASOS)      # se asienta: ya no se va hacia los lados, pero se hunde
                Xp = X + V * dt
                for _ in range(s.ITERACIONES):
                    # 1) cada pedacito de carne vuelve un poco a su forma (girado como esté ahora)
                    Xc = Xp[Cs]
                    c = (Xc * mc[..., None]).sum(1) / Mc[:, None]
                    A = np.einsum('nk,nka,nkb->nab', mc, Xc - c[:, None], Q0)
                    U, _, Vt = np.linalg.svd(A)
                    d = np.sign(np.linalg.det(U @ Vt)); U[:, :, 2] *= d[:, None]
                    R = U @ Vt
                    meta = c[:, None] + np.einsum('nab,nkb->nka', R, Q0)
                    acc = np.zeros_like(Xp)
                    np.add.at(acc, Cs[s.Cm], meta[s.Cm])
                    Xp += (s.FORMA * s.forma_p)[:, None] * (acc / np.maximum(cuenta, 1)[:, None] - Xp)
                    # 2) la carne casi no se estira
                    dd = Xp[b] - Xp[a]; L = np.linalg.norm(dd, axis=1) + 1e-9
                    err = np.where(L > s.largo * 1.03, L - s.largo * 1.03,
                                   np.where(s.a_lo_largo & (L < s.largo * 0.9), L - s.largo * 0.9, 0.0))
                    kk = np.where(err > 0, s.ESTIRA, s.LARGO)
                    corr = (kk * err / (L * (wa + wb)))[:, None] * dd
                    dx = np.zeros_like(Xp)
                    np.add.at(dx, a, corr * wa[:, None]); np.add.at(dx, b, -corr * wb[:, None])
                    Xp += dx * 0.5
                    # 3) cabeza rígida
                    if len(s.ic):
                        cc = Xp[s.ic].mean(0); Rc = _polar((Xp[s.ic] - cc).T @ s.Xc0)
                        if s.CABEZA_ARRIBA is not None:
                            Rc = s._cabeza_derecha(Rc, float(np.clip((f / s.FPS - s.CABEZA_DESDE) / s.CABEZA_TARDA, 0, 1)))
                        Xp[s.ic] = cc + s.Xc0 @ Rc.T
                    # 4) piso
                    Xp[:, 2] = np.maximum(Xp[:, 2], s.radio)
                # 5) el cuerpo no se atraviesa a sí mismo (partículas que no son vecinas)
                pp = cKDTree(Xp).query_pairs(2 * s.radio, output_type='ndarray')
                if len(pp):
                    pp = np.array([q for q in pp if (min(q), max(q)) not in vecino], int).reshape(-1, 2)
                if len(pp):
                    dd = Xp[pp[:, 1]] - Xp[pp[:, 0]]; L = np.linalg.norm(dd, axis=1) + 1e-9
                    c_ = (0.5 * (2 * s.radio - L) / L)[:, None] * dd
                    np.add.at(Xp, pp[:, 0], -c_ * 0.5); np.add.at(Xp, pp[:, 1], c_ * 0.5)
                Vn = (Xp - X) / dt
                suelo = Xp[:, 2] <= s.radio + 1e-4
                Vn[suelo, :2] *= (1 - s.ROCE)
                Vn[suelo, 2] = 0.0
                Vn[:, 2] = np.minimum(Vn[:, 2], s.SUBE_MAX if f > 3 else 1e9)     # nada sube de golpe: sin rebote
                # lo doblado se queda doblado: la forma de reposo de cada pedacito se acerca a la de ahora
                if s.PLASTICO > 0:
                    Xc = Xp[Cs]; c = (Xc * mc[..., None]).sum(1) / Mc[:, None]
                    A = np.einsum('nk,nka,nkb->nab', mc, Xc - c[:, None], Q0)
                    U, _, Vt = np.linalg.svd(A); d = np.sign(np.linalg.det(U @ Vt)); U[:, :, 2] *= d[:, None]
                    R = U @ Vt
                    ahora = np.einsum('nba,nkb->nka', R, Xc - c[:, None]) * Cm[..., None]   # en el giro del pedacito
                    Q0 = Q0 + s.PLASTICO * (ahora - Q0)
                X, V = Xp, Vn
            frames.append(X.copy())
        s.frames = np.array(frames)

    def _cabeza_derecha(s, Rc, u):
        """Lleva el giro de la cabeza (Rc) hacia derecha, mirando CABEZA_ARRIBA grados hacia arriba (u: 0..1)."""
        from scipy.spatial.transform import Rotation as Ro, Slerp
        z = np.array([0, 0, 1.0])
        fc = Rc @ s.cab_marco0[:, 0]; fh = fc - z * (fc @ z)
        if np.linalg.norm(fh) < 1e-3:
            return Rc
        fh /= np.linalg.norm(fh); a = np.radians(s.CABEZA_ARRIBA)
        ft = np.cos(a) * fh + np.sin(a) * z; ut = -np.sin(a) * fh + np.cos(a) * z
        Rt = np.stack([ft, ut, np.cross(ft, ut)], 1) @ s.cab_marco0.T
        w = u * u * (3 - 2 * u)
        return Slerp([0, 1], Ro.from_matrix([Rc, Rt]))(w).as_matrix()

    def pose(s, t):
        """Como m.pose(clip, t): antes de t0 es Max normal; después, el cuerpo blando."""
        if t < s.t0:
            return s.m.pose(s.clip, t)
        f = min(int(round((t - s.t0) * s.FPS)), len(s.frames) - 1)
        X = s.frames[f]
        R = s._rotaciones(X)
        res = []
        for p, V0, (j, w) in zip(s.m.prims, s.V0, s.atado):
            V = np.einsum('vk,vki->vi', w, X[j] + np.einsum('vkab,vkb->vka', R[j], V0[:, None, :] - s.X0[j]))
            F = p['idx'].reshape(-1, 3) if len(p['idx']) else np.zeros((0, 3), int)
            Nn = _normales(V, F) if len(F) else np.zeros_like(V)
            res.append((V[p['idx']], Nn[p['idx']]))
        return res
