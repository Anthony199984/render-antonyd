"""ZIP 37 · Max V14 = Max V13 (ZIP 36) + LAS MEJORAS DE LA LISTA (30 oct 2026). Archivo NUEVO: no cambia max_v13.py.

Desde el ZIP 37 se dibuja con MaxCaraV14 (y PistaCaraV14):
    from soma.max_v14 import MaxCaraV14, PistaCaraV14
    m = MaxCaraV14(glb='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV14.desde_escena('<carpeta de salida de escena5.py o escena6.py>', '<escena.json>')
    res = m.pose(clip, t)

Trae todo lo de Max V13 (sin tirones, brazo abierto al agarrar, codo estable, codo sin saltos) y además:
 1. MUÑECA RECTA AL AGARRAR (MUNECA_RECTA): con la mano fija, el codo gira lo justo para que la muñeca quede menos
    doblada, sin cerrar el brazo (el codo sigue hacia afuera). Ver _CodoSinSaltos.
 2. (la revisión con objetos va aparte: soma/objetos_escena.py, escena6.py y revision_max.py)
 3. POSTURA (ERGUIDO): tronco más derecho y cabeza menos colgada, quieto o caminando con las manos libres. Ver _Erguido.
 4. MANO CON TONO (MANO_TONO): la mano libre ya no cuelga doblada; sigue más la línea del antebrazo. Ver _Erguido.
 5. BORDE DE LA MANGA LIMPIO (MANGA_LIMPIA): corte limpio, con grosor, sin dientes ni doble filo. Ver _MangaLimpia.
 6. SOLTAR SUAVE (SUELTA_SUAVE): al soltar un objeto el brazo sale suave, sin caer de golpe. Ver _SueltaSuave.
Mientras una mano SOSTIENE un objeto no se mueve (0 mm con Max V13) y los pies tampoco.
Cada mejora se apaga sola: MaxCaraV14.<NOMBRE> = False.
"""
import numpy as np

from .max_v12 import MaxCaraV12, MaxV12, PistaCaraV12

class PistaCaraV14(PistaCaraV12):
    """PistaCaraV12 + los agarres de la escena (de *_informe.json): cuándo cada mano sostiene un objeto."""

    @classmethod
    def desde_escena(cls, carpeta, escena_json=None, semilla=7):
        p = super().desde_escena(carpeta, escena_json, semilla)
        p.__class__ = cls
        p.agarres_escena = agarres_de_escena(carpeta)
        return p


def agarres_de_escena(carpeta):
    """[(segundo_agarra, segundo_suelta, 'Left'/'Right')] del informe de escena5.py."""
    import json
    from pathlib import Path
    inf = next(Path(carpeta).glob("*_informe.json"), None)
    if inf is None:
        return None
    out = []
    for a in json.loads(inf.read_text(encoding="utf-8")).get("agarres", []) or []:
        if a.get("segundo") is None:
            continue
        t0 = float(a["segundo"]); t1 = t0 + float(a.get("sostener") or a.get("cuadros_agarrando", 60) / 30.0)
        lados = ('Left', 'Right') if a.get("mano") in ("dos", "ambas", "las dos") else \
            (('Left',) if a.get("mano") == "izq" else ('Right',))
        for L in lados:
            out.append((t0, t1, L))
    return out


class _SinTirones:
    SIN_TIRONES = True
    TIRON_MIN = 6.0          # grados por cuadro (a 30 fps): por debajo de esto nunca es un tirón
    TIRON_VECES = 2.5        # el cuadro tiene que ser así de veces más rápido que sus vecinos
    TIRON_SIGMA = 4.0        # cuadros de alisado en el tirón (un salto de 30° queda repartido en ~10 cuadros)
    TIRON_ANCHO = 10         # cuadros antes y después del tirón que se alisan (los 3 del centro del todo, luego se apaga suave)
    # grupos que se revisan juntos: si un hueso del grupo da el tirón, se alisa todo el grupo (el brazo entero)
    TIRON_GRUPOS = {
        'brazo_izq': ('LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'),
        'brazo_der': ('RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'),
        'tronco': ('Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head'),
    }

    def _sin_tirones(s, clip):
        if not s.SIN_TIRONES:
            return clip
        c = s.__dict__.setdefault('_clips_sin_tirones', {})
        if id(clip) in c:
            return c[id(clip)][1]
        import copy
        from scipy.spatial.transform import Rotation as Ro
        from scipy.ndimage import median_filter
        n = clip.n
        L = clip.L.copy()
        s.tirones = []
        # mano LIBRE por cuadro (como en _SinTemblor: dedos en su pose natural). Con la mano agarrando no se alisa nada:
        # el objeto va pegado a esa mano tal como lo dejó la escena, y alisar el brazo los separaría.
        import os
        tabla = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_v7_mano_natural.npz'))
        libre = {}
        agarres = getattr(getattr(s, 'pista', None), 'agarres_escena', None)
        if agarres is not None:
            # la escena dice cuándo agarra cada mano (más seguro que mirar los dedos: un puño cerrado no es un agarre)
            tt = np.arange(n) / clip.fps
            for lado in ('Left', 'Right'):
                lb = np.ones(n)
                for t0, t1, mano in agarres:
                    if mano == lado:
                        lb = np.minimum(lb, np.clip(np.maximum((t0 - 0.35 - tt) / 0.2, (tt - t1 - 0.05) / 0.1), 0, 1))
                libre[lado] = lb
        for lado in ('Left', 'Right'):
            if lado in libre:
                continue
            dev = np.zeros(n)
            for nom, D in zip(tabla['nombres'], tabla['defecto']):
                nom = str(nom)
                if nom.startswith(lado) and nom in clip.idx:
                    R_ = np.einsum('nji,jk->nik', clip.L[:, clip.idx[nom]], D)
                    dev = np.maximum(dev, Ro.from_matrix(R_).magnitude())
            lb = np.clip(1 - dev / 0.25, 0, 1)
            for _ in range(10):                                         # se apaga antes y después del agarre
                lb = np.minimum(lb, np.minimum(np.r_[lb[0], lb[:-1]], np.r_[lb[1:], lb[-1]]))
            libre[lado] = lb
        libre_grupo = {'brazo_izq': libre['Left'], 'brazo_der': libre['Right'],
                       'tronco': np.minimum(libre['Left'], libre['Right'])}
        k = np.arange(-int(3 * s.TIRON_SIGMA), int(3 * s.TIRON_SIGMA) + 1)
        wk = np.exp(-0.5 * (k / s.TIRON_SIGMA) ** 2); wk /= wk.sum()
        for grupo, huesos in s.TIRON_GRUPOS.items():
            js = [clip.idx[h] for h in huesos if h in clip.idx]
            if not js or n < 8:
                continue
            mascara = np.zeros(n)
            for j in js:
                R = Ro.from_matrix(clip.L[:, j])
                w = np.r_[0.0, np.degrees((R[1:] * R[:-1].inv()).magnitude())]       # grados por cuadro
                # rapidez "normal" alrededor (sin contar el propio cuadro ni los de al lado)
                ref = np.zeros(n)
                for f in range(n):
                    vec = np.r_[w[max(0, f - 5):max(0, f - 1)], w[f + 2:f + 6]]
                    ref[f] = np.median(vec) if len(vec) else 0.0
                lg = libre_grupo.get(grupo, np.ones(n))
                pic = np.where((w > s.TIRON_MIN) & (w > s.TIRON_VECES * ref + 1.0) & (lg > 0.5))[0]
                for f in pic:
                    s.tirones.append((int(f), grupo, round(float(w[f]), 1)))
                    for d in range(-s.TIRON_ANCHO, s.TIRON_ANCHO + 1):
                        g = f + d
                        if 0 <= g < n:
                            u = max(0.0, (abs(d) - 3) / (s.TIRON_ANCHO - 2))
                            mascara[g] = max(mascara[g], 0.5 + 0.5 * np.cos(np.pi * min(u, 1.0)))
            mascara *= libre_grupo.get(grupo, np.ones(n))
            if mascara.max() <= 0:
                continue
            for j in js:
                q = Ro.from_matrix(clip.L[:, j]).as_quat()
                for f in range(1, n):
                    if q[f] @ q[f - 1] < 0:
                        q[f] = -q[f]
                qs = np.zeros_like(q)
                for kk, w_ in zip(k, wk):
                    qs += w_ * q[np.clip(np.arange(n) + kk, 0, n - 1)]
                qs /= np.linalg.norm(qs, axis=1, keepdims=True)
                qf = q * (1 - mascara[:, None]) + qs * mascara[:, None]
                qf /= np.linalg.norm(qf, axis=1, keepdims=True)
                L[:, j] = Ro.from_quat(qf).as_matrix()
        s.tirones.sort()
        if not s.tirones:
            c[id(clip)] = (clip, clip)
            return clip
        nuevo = copy.copy(clip)
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
        return super().globales(s._sin_tirones(clip), t, bucle, pies)


class _CodoSinSaltos:
    """El codo no "salta" de lado. Para llevar la mano a su sitio (agarres) Max acomoda el brazo con IK y pone el codo
    hacia donde lo tenía la animación; cuando el brazo está casi estirado (por ejemplo al alcanzar una caja) ese lado
    casi no se distingue y el codo podía girar 30-90° de un cuadro al siguiente alrededor de la línea hombro-mano.
    Se revisa la animación entera (30 cuadros por segundo): donde el codo gira de golpe, ese giro se reparte en unos
    cuadros antes y después. La mano NO se mueve (el giro es alrededor de la línea hombro-mano) ni cambia su giro."""
    CODO_SIN_SALTOS = True
    CODO_SALTO_MIN = 12.0    # grados en un cuadro (a 30 fps) para contar como salto
    CODO_SALTO_VECES = 3.0   # y tantas veces más rápido que los cuadros de alrededor
    CODO_SIGMA = 3.0         # cuadros de reparto
    CODO_ANCHO = 8           # cuadros antes y después
    CODO_ALISADO = 2.5       # cuadros de alisado suave del giro del codo en toda la animación (0 = no)
    # MUÑECA RECTA AL AGARRAR: con la mano fija (sitio y giro), el codo gira alrededor de la línea hombro-mano lo justo
    # para que el antebrazo quede lo más alineado posible con la mano (muñeca menos doblada), sin cerrar el brazo
    # (el codo sigue hacia afuera) y cambiando suave de un cuadro al otro.
    MUNECA_RECTA = True
    MUNECA_GIRO_MAX = 50.0   # grados que puede girar el codo para enderezar la muñeca
    MUNECA_AFUERA_MIN = 0.6  # el codo nunca menos afuera que esto (0 = de lado, 1 = del todo afuera): brazo ABIERTO

    def _brazo_ids(s, L):
        ix = s.idx
        return ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']

    def _plan_codo13(s, clip, bucle):
        c = s.__dict__.setdefault('_planes_codo13', {})
        if (id(clip), bucle) in c:
            return c[(id(clip), bucle)][1]

        def calcular(cl):
            from scipy.ndimage import gaussian_filter1d
            n = int(np.floor(cl.dur * 30)) + 1
            th = {'Left': np.zeros(n), 'Right': np.zeros(n)}
            gm = {'Left': np.zeros(n), 'Right': np.zeros(n)}; wm = {'Left': np.zeros(n), 'Right': np.zeros(n)}
            prev = {}
            gs = np.radians(np.arange(-s.MUNECA_GIRO_MAX, s.MUNECA_GIRO_MAX + 0.1, 2.5))
            for f in range(n):
                G, Pm, b = super(_CodoSinSaltos, s).globales(cl, f / 30.0, bucle, True)
                ix = s.idx
                izq = Pm[ix['LeftArm']] - Pm[ix['RightArm']]; izq /= np.linalg.norm(izq) + 1e-12
                for L in ('Left', 'Right'):
                    iA, iF, iH = s._brazo_ids(L)
                    A, E, H = Pm[iA], Pm[iF], Pm[iH]
                    e = (H - A) / (np.linalg.norm(H - A) + 1e-12)
                    p = (E - A) - e * ((E - A) @ e)
                    # muñeca recta (solo mientras va a agarrar y sostiene)
                    w_ag = s._peso_agarre(L, f / 30.0) if (s.MUNECA_RECTA and hasattr(s, '_peso_agarre')) else 0.0
                    if w_ag > 1e-3 and (L + 'HandMiddle1') in ix and np.linalg.norm(p) > 0.01:
                        hd = Pm[ix[L + 'HandMiddle1']] - H; hd /= np.linalg.norm(hd) + 1e-12
                        afuera = izq if L == 'Left' else -izq
                        K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
                        mejor, gb = None, 0.0
                        for g in gs:
                            Rg = np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K
                            E2 = A + Rg @ (E - A); fa = (H - E2) / (np.linalg.norm(H - E2) + 1e-12)
                            pp = Rg @ p; af = (pp / np.linalg.norm(pp)) @ afuera
                            c_ = np.degrees(np.arccos(np.clip(fa @ hd, -1, 1))) + 0.15 * abs(np.degrees(g))
                            c_ += 400 * max(0.0, s.MUNECA_AFUERA_MIN - af)
                            if mejor is None or c_ < mejor:
                                mejor, gb = c_, g
                        gm[L][f] = gb; wm[L][f] = w_ag
                    if L in prev and np.linalg.norm(p) > 0.02 and np.linalg.norm(prev[L]) > 0.02:
                        q = prev[L] - e * (prev[L] @ e)
                        d = np.arctan2(np.cross(q, p) @ e, q @ p)
                        th[L][f] = th[L][f - 1] + d
                    elif f > 0:
                        th[L][f] = th[L][f - 1]
                    prev[L] = p
            plan = {}
            k = np.arange(-int(3 * s.CODO_SIGMA), int(3 * s.CODO_SIGMA) + 1)
            for L in ('Left', 'Right'):
                x = th[L]; w = np.degrees(np.abs(np.r_[0.0, np.diff(x)]))
                masc = np.zeros(n); saltos = []
                for f in range(1, n):
                    vec = np.r_[w[max(0, f - 6):max(0, f - 1)], w[f + 2:f + 7]]
                    ref = np.median(vec) if len(vec) else 0.0
                    if w[f] > s.CODO_SALTO_MIN and w[f] > s.CODO_SALTO_VECES * ref + 1.0:
                        saltos.append((f, round(float(w[f]), 1)))
                        for dd in range(-s.CODO_ANCHO, s.CODO_ANCHO + 1):
                            g = f + dd
                            if 0 <= g < n:
                                u = max(0.0, (abs(dd) - 2) / (s.CODO_ANCHO - 1))
                                masc[g] = max(masc[g], 0.5 + 0.5 * np.cos(np.pi * min(u, 1.0)))
                xs = gaussian_filter1d(x, s.CODO_SIGMA, mode='nearest')
                # además, en TODA la animación el giro del codo se alisa un poco (CODO_ALISADO cuadros: no se nota en un
                # movimiento normal y quita los giros rápidos que no llegan a contar como salto)
                xa = gaussian_filter1d(x, s.CODO_ALISADO, mode='nearest') if s.CODO_ALISADO > 0 else x
                dmun = gaussian_filter1d(gm[L] * wm[L], 3.0, mode='nearest')          # muñeca recta, suave
                plan[L] = ((xs - x) * masc + (xa - x) * (1 - masc) + dmun, saltos)
            return plan

        plan = s._con_disco('codo14', clip, calcular) if hasattr(s, '_con_disco') else calcular(clip)
        c[(id(clip), bucle)] = (clip, plan)
        return plan

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if not s.CODO_SIN_SALTOS:
            return G, Pm, b
        plan = s._plan_codo13(clip, bucle)
        s.codo_saltos = {L: plan[L][1] for L in plan if L in ('Left', 'Right')}
        f = float(t) * 30.0
        for L in ('Left', 'Right'):
            dl = plan[L][0]
            if not len(dl):
                continue
            i0 = int(np.clip(np.floor(f), 0, len(dl) - 1)); i1 = min(i0 + 1, len(dl) - 1); w1 = float(np.clip(f - i0, 0, 1))
            g = dl[i0] * (1 - w1) + dl[i1] * w1
            if abs(g) < np.radians(0.2):
                continue
            iA, iF, iH = s._brazo_ids(L)
            A, H = Pm[iA].copy(), Pm[iH].copy()
            e = (H - A) / (np.linalg.norm(H - A) + 1e-12)
            K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            R = np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K
            G = G.copy(); Pm = Pm.copy()
            G[iA] = R @ G[iA]; G[iF] = R @ G[iF]           # brazo y antebrazo giran; la mano y los dedos quedan igual
            Pm[iF] = A + R @ (Pm[iF] - A)
        return G, Pm, b


class _CodoEstable:
    """La CAUSA de los saltos del codo: para llevar la mano a su sitio, Max acomoda el brazo (IK) y pone el codo hacia el
    mismo lado que el brazo de referencia. Cuando ese brazo de referencia está casi ESTIRADO (codo doblado menos de
    ~1-2 cm, por ejemplo al alcanzar una caja) ese "lado" no existe: es ruido, y el codo de Max giraba de golpe 100-150°.
    Ahora, cuanto más estirado está el brazo de referencia, más se usa el lado natural del codo (hacia abajo y un poco
    afuera, como cuelga un brazo); con el brazo bien doblado se usa el de siempre. La mano queda en el mismo sitio."""
    CODO_ESTABLE = True
    CODO_ESTIRADO = 0.015    # m: con el codo de referencia a menos de esto de la línea hombro-mano se usa el lado natural
    CODO_DOBLADO = 0.15      # m: desde aquí se usa el lado de la referencia (en medio, una mezcla suave)
    # BRAZO ABIERTO AL AGARRAR: mientras la mano va hacia un objeto y lo sostiene, el brazo se abre (codo hacia afuera,
    # separado del cuerpo, un poco hacia abajo) en vez de meterse hacia el cuerpo; la mano no se mueve ni cambia su giro.
    CODO_NATURAL_AGARRE = 0.85   # 0 = como lo dejó agarres_ik, 1 = del todo hacia el lado del brazo abierto
    CODO_AGARRE_ABAJO = 0.35     # el codo va hacia AFUERA (brazo abierto, separado del cuerpo) y esto hacia abajo
    CODO_NATURAL_ANTES = 1.5     # s antes de agarrar en que empieza (cuando la mano va hacia el objeto)

    def _manos_como_antes(s, clip, t, bucle, G, Pm, b):
        s._pm_pre = Pm.copy() if s.CODO_ESTABLE else None
        try:
            return super()._manos_como_antes(clip, t, bucle, G, Pm, b)
        finally:
            s._pm_pre = None

    def _posiciones_maxsoma(s, clip, t, bucle):
        P0 = super()._posiciones_maxsoma(clip, t, bucle)
        Pm = getattr(s, '_pm_pre', None)
        if Pm is None:
            return P0
        ix = s.idx
        P0 = np.array(P0, dtype=float).copy()
        izq = Pm[ix['LeftArm']] - Pm[ix['RightArm']]; izq /= np.linalg.norm(izq) + 1e-12
        arriba = Pm[ix['Neck1']] - Pm[ix['Spine2']] if 'Neck1' in ix else Pm[ix['Head']] - Pm[ix['Hips']]
        arriba = arriba - izq * (arriba @ izq); arriba /= np.linalg.norm(arriba) + 1e-12
        s.codo_natural = {}
        for L, afuera in (('Left', izq), ('Right', -izq)):
            iA, iF, iH = ix[L + 'Arm'], ix[L + 'ForeArm'], ix[L + 'Hand']
            A0, E0, H0 = P0[iA], P0[iF], P0[iH]
            e = H0 - A0; ln = np.linalg.norm(e)
            if ln < 1e-6:
                continue
            e = e / ln
            rp = (E0 - A0) - e * ((E0 - A0) @ e); r = np.linalg.norm(rp)
            nat = -arriba + 0.5 * afuera; nat = nat - e * (nat @ e)
            if np.linalg.norm(nat) < 1e-6:
                continue
            nat /= np.linalg.norm(nat)
            u = np.clip((r - s.CODO_ESTIRADO) / (s.CODO_DOBLADO - s.CODO_ESTIRADO), 0, 1); w = u * u * (3 - 2 * u)
            if w >= 0.999 and s._peso_agarre(L, t) <= 1e-3:
                continue
            if r > 1e-9:
                ru = rp / r
                ang = np.arctan2(np.cross(nat, ru) @ e, nat @ ru)            # de natural hacia la referencia
                g = ang * w
                K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
                d = (np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K) @ nat
            else:
                d = nat
            # codo natural mientras la mano va hacia el objeto y lo sostiene
            a_ag = s._peso_agarre(L, t)
            if a_ag > 1e-3 and s.CODO_NATURAL_AGARRE > 0:
                nat2 = s.CODO_AGARRE_ABAJO * (-arriba) + afuera                  # brazo ABIERTO: codo hacia afuera
                nat2 /= np.linalg.norm(nat2); nat2 = nat2 - e * (nat2 @ e); q2 = np.linalg.norm(nat2)
                # con el brazo colgando derecho hacia abajo "abajo" no dice nada: ahí no se toca (entra suave)
                u2 = np.clip((q2 - 0.3) / 0.35, 0, 1); f2 = u2 * u2 * (3 - 2 * u2)
                if f2 > 1e-3:
                    nat2 /= q2
                    ang2 = np.arctan2(np.cross(d, nat2) @ e, d @ nat2)
                    g2 = ang2 * a_ag * s.CODO_NATURAL_AGARRE * f2
                    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
                    d = (np.eye(3) + np.sin(g2) * K + (1 - np.cos(g2)) * K @ K) @ d
            largo = max(r, 0.03)                                           # basta con que el lado quede claro
            P0[iF] = A0 + e * ((E0 - A0) @ e) + d * largo
            s.codo_natural[L] = round(1 - w, 2)
        return P0

    def _peso_agarre(s, L, t):
        """0..1: la mano L va hacia un objeto o lo sostiene (según los agarres de la escena)."""
        ag = getattr(getattr(s, 'pista', None), 'agarres_escena', None) or []
        best = 0.0
        for t0, t1, mano in ag:
            if mano != L:
                continue
            sube = np.clip((t - (t0 - s.CODO_NATURAL_ANTES)) / 0.5, 0, 1)
            baja = np.clip((t1 + 0.6 - t) / 0.5, 0, 1)
            x = min(sube, baja); best = max(best, x * x * (3 - 2 * x))
        return best


class _Erguido:
    """POSTURA (mejora 3): Kimodo deja a Max algo encorvado (el tronco ~8-10° hacia adelante, quieto y caminando). Aquí
    el tronco se endereza: se quita parte de esa inclinación repartida en la columna (la cadera y los pies no se
    mueven), y la cabeza acompaña a medias (sigue mirando casi a donde miraba). No se toca cuando se agacha a propósito
    (más de ~20°) ni mientras una mano va a agarrar o sostiene algo (el agarre queda igual).
    MANO CON TONO (mejora 4): la mano LIBRE que cuelga muy doblada en la muñeca (mano "floja") se endereza un poco
    hacia la línea del antebrazo; la mano que va a agarrar o sostiene no se toca."""
    ERGUIDO = True
    ERGUIR = 0.8              # parte de la inclinación de más que se quita
    ERGUIR_OBJETIVO = 3.0     # grados de inclinación natural que se dejan
    ERGUIR_AGACHA = 20.0      # desde aquí es que se agacha a propósito: se deja de tocar (hasta 30° se apaga suave)
    CUELLO_SIGUE = 0.5        # la cabeza se endereza la mitad de lo que se endereza el tronco
    CABEZA_ERGUIR = 0.7       # cabeza colgando hacia abajo sin mirar nada: se levanta esta parte de lo que pasa de...
    CABEZA_OBJETIVO = 10.0    # ...estos grados (no se toca mientras mira lo que va a agarrar o sostiene)
    MANO_TONO = True
    MANO_TONO_OBJ = 15.0      # grados de doblez de muñeca que se dejan en una mano libre
    MANO_TONO_QUITA = 0.6     # parte del doblez de más que se quita

    def _peso_libre(s, n, fps):
        tt = np.arange(n) / fps
        w = {}
        for L in ('Left', 'Right'):
            w[L] = np.array([1.0 - s._peso_agarre(L, t) for t in tt]) if hasattr(s, '_peso_agarre') else np.ones(n)
        return w

    def _erguir(s, clip):
        if not (s.ERGUIDO or s.MANO_TONO):
            return clip
        c = s.__dict__.setdefault('_clips_erguidos', {})
        if id(clip) in c:
            return c[id(clip)][1]
        import copy
        from scipy.ndimage import gaussian_filter1d
        n = clip.n; ix = clip.idx
        need = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'LeftArm', 'RightArm')
        if not all(k in ix for k in need):
            c[id(clip)] = (clip, clip); return clip
        G, P, L = clip.G, clip.P, clip.L.copy()
        libre = s._peso_libre(n, clip.fps)
        ambas = np.minimum(libre['Left'], libre['Right'])
        up = np.array([0, 1.0, 0])
        lat = P[:, ix['LeftArm']] - P[:, ix['RightArm']]; lat[:, 1] = 0
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-12
        fwd = np.cross(lat, up)
        v = P[:, ix['Neck1']] - P[:, ix['Hips']]
        incl = np.degrees(np.arctan2((v * fwd).sum(1), (v * up).sum(1)))
        def suave(x, a, b):
            u = np.clip((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u)
        corr = np.zeros(n)
        if s.ERGUIDO:
            corr = -s.ERGUIR * np.maximum(incl - s.ERGUIR_OBJETIVO, 0) * (1 - suave(incl, s.ERGUIR_AGACHA, s.ERGUIR_AGACHA + 10))
            corr = gaussian_filter1d(corr, 6.0, mode='nearest') * gaussian_filter1d(ambas, 3.0, mode='nearest')
        s.erguido_grados = corr
        # cabeza colgando (sin contar lo que ya se enderezó con el tronco)
        corr_c = np.zeros(n)
        if s.ERGUIDO and 'HeadEnd' in ix and 'Neck2' in ix:
            hv = P[:, ix['HeadEnd']] - P[:, ix['Neck1']]
            cab = np.degrees(np.arctan2((hv * fwd).sum(1), (hv * up).sum(1))) + corr * s.CUELLO_SIGUE
            corr_c = -s.CABEZA_ERGUIR * np.maximum(cab - s.CABEZA_OBJETIVO, 0)
            corr_c = gaussian_filter1d(corr_c, 6.0, mode='nearest') * gaussian_filter1d(ambas, 3.0, mode='nearest')
        s.cabeza_erguida_grados = corr_c
        def rot(e, g):
            K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            return np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K
        reparto = (('Spine1', 0.35), ('Spine2', 0.35), ('Chest', 0.30))
        for f in range(n):
            g = np.radians(corr[f])
            if abs(g) > 1e-4 or abs(corr_c[f]) > 1e-4:
                # el eje "a lo ancho" del cuerpo; girar hacia atrás = enderezar
                for nom, fr in reparto:
                    j = ix[nom]; pa = clip.padres[j]
                    L[f, j] = G[f, pa].T @ rot(lat[f], g * fr) @ G[f, j]          # g < 0: hacia atrás
                j = ix['Neck1']; pa = clip.padres[j]
                L[f, j] = G[f, pa].T @ rot(lat[f], -g * (1 - s.CUELLO_SIGUE)) @ G[f, j]
            gc = np.radians(corr_c[f])
            if abs(gc) > 1e-4:
                for nom, fr in (('Neck1', 0.5), ('Neck2', 0.5)):
                    j = ix[nom]; pa = clip.padres[j]
                    L[f, j] = (G[f, pa].T @ rot(lat[f], gc * fr) @ G[f, pa]) @ L[f, j]   # se suma a lo que ya tenga
        # mano con tono (sobre las poses originales: la cadena de arriba no cambia la relación antebrazo-mano)
        s.mano_tono_grados = {'Left': np.zeros(n), 'Right': np.zeros(n)}
        if s.MANO_TONO:
            for lado in ('Left', 'Right'):
                if not all(k in ix for k in (lado + 'ForeArm', lado + 'Hand', lado + 'HandMiddle1')):
                    continue
                jf, jh, jm = ix[lado + 'ForeArm'], ix[lado + 'Hand'], ix[lado + 'HandMiddle1']
                # con tono también mientras la mano VA hacia el objeto; se suelta suave en los últimos 0,25 s antes de
                # tocarlo (ahí manda el agarre) y no se toca mientras sostiene
                ag = getattr(getattr(s, 'pista', None), 'agarres_escena', None)
                if ag is not None:
                    tt = np.arange(n) / clip.fps; w = np.ones(n)
                    for t0_, t1_, mano in ag:
                        if mano == lado:
                            w = np.minimum(w, np.clip(np.maximum((t0_ - 0.25 - tt) / 0.3, (tt - t1_ - 0.3) / 0.3), 0, 1))
                else:
                    w = libre[lado]
                w = gaussian_filter1d(w, 2.0, mode='nearest')
                for f in range(n):
                    if w[f] < 1e-3:
                        continue
                    a = P[f, jh] - P[f, jf]; a /= np.linalg.norm(a) + 1e-12
                    h = P[f, jm] - P[f, jh]; h /= np.linalg.norm(h) + 1e-12
                    b = np.degrees(np.arccos(np.clip(a @ h, -1, 1)))
                    if b <= s.MANO_TONO_OBJ:
                        continue
                    eje = np.cross(h, a)
                    if np.linalg.norm(eje) < 1e-6:
                        continue
                    eje /= np.linalg.norm(eje)
                    q = np.radians((b - s.MANO_TONO_OBJ) * s.MANO_TONO_QUITA * w[f])
                    L[f, jh] = G[f, jf].T @ rot(eje, q) @ G[f, jh]
                    s.mano_tono_grados[lado][f] = np.degrees(q)
        if not np.any(np.abs(corr) > 1e-4) and not np.any(np.abs(corr_c) > 1e-4) and \
                not any(v_.any() for v_ in s.mano_tono_grados.values()):
            c[id(clip)] = (clip, clip); return clip
        nuevo = copy.copy(clip)
        Gn = np.zeros_like(clip.G); Pn = np.zeros_like(clip.P)
        for j in range(len(clip.nombres)):
            pa = clip.padres[j]
            if pa < 0:
                Gn[:, j] = L[:, j]; Pn[:, j] = clip.T[:, j]
            else:
                Gn[:, j] = Gn[:, pa] @ L[:, j]; Pn[:, j] = Pn[:, pa] + np.einsum('nij,nj->ni', Gn[:, pa], clip.T[:, j])
        nuevo.L, nuevo.G, nuevo.P = L, Gn, Pn
        if hasattr(nuevo, '_contactos'):
            nuevo._contactos = None
        c[id(clip)] = (clip, nuevo)
        return nuevo

    def globales(s, clip, t, bucle=False, pies=True):
        return super().globales(s._erguir(clip), t, bucle, pies)


class _SueltaSuave:
    """AL SOLTAR (mejora 6): en la escena, al soltar el objeto la mano pasaba de golpe a lo que hacía la animación (el
    brazo caía a un lado o subía en 0,2-0,3 s, hasta 3-7 m/s). Ahora el brazo sale del objeto suave: desde la pose en que
    soltó va pasando poco a poco (SUELTA_SEGUNDOS) a la animación, sin tirón al empezar ni al terminar. Mientras
    sostiene no se toca nada (el agarre queda igual)."""
    SUELTA_SUAVE = True
    SUELTA_SEGUNDOS = 0.8      # lo que tarda el brazo en pasar de la pose de soltar a la animación
    SUELTA_EMPIEZA = 0.4       # m/s: la mano empieza a irse cuando pasa de esto (después del segundo de soltar)
    SUELTA_HUESOS = ('Shoulder', 'Arm', 'ForeArm', 'Hand')
    SUELTA_LIBRE_SUAVE = True  # lo de la mano libre (brazo relajado, abierto) entra al mismo paso suave

    def _soltar(s, clip):
        if not s.SUELTA_SUAVE:
            return clip
        c = s.__dict__.setdefault('_clips_sueltos', {})
        if id(clip) in c:
            return c[id(clip)][1]
        ag = getattr(getattr(s, 'pista', None), 'agarres_escena', None) or []
        if not ag:
            c[id(clip)] = (clip, clip); return clip
        import copy
        from scipy.spatial.transform import Rotation as Ro, Slerp
        n, fps, ix = clip.n, clip.fps, clip.idx
        L = clip.L.copy()
        s.sueltas = []; s._ventanas_suelta = {}
        hechos = set()
        for t0, t1, lado in sorted(ag):
            if (round(t1, 3), lado) in hechos or lado + 'Hand' not in ix:
                continue
            hechos.add((round(t1, 3), lado))
            h = clip.P[:, ix[lado + 'Hand']]
            v = np.r_[0.0, np.linalg.norm(np.diff(h, axis=0), axis=1) * fps]
            f1 = int(round(t1 * fps))
            if f1 >= n - 2:
                continue
            fs = next((f for f in range(max(1, f1 - 3), min(n, f1 + 20)) if v[f] > s.SUELTA_EMPIEZA), None)
            if fs is None:
                continue
            fr = fs - 1                                            # última pose quieta, todavía en el objeto
            N = int(round(s.SUELTA_SEGUNDOS * fps))
            # no se mete en el próximo agarre de esa misma mano
            prox = [a for a, b, m_ in ag if m_ == lado and a > t1]
            if prox:
                N = min(N, int((min(prox) - 0.6) * fps) - fr)
            N = min(N, n - 1 - fr)
            if N < 4:
                continue
            u = np.arange(N + 1) / N
            w = u ** 3 * (u * (u * 6 - 15) + 10)                   # suave al empezar y al terminar
            for nom in s.SUELTA_HUESOS:
                if lado + nom not in ix:
                    continue
                j = ix[lado + nom]
                for i in range(1, N + 1):
                    f = fr + i
                    r = Ro.from_matrix(np.stack([L[fr, j], L[f, j]]))
                    L[f, j] = Slerp([0, 1], r)([w[i]]).as_matrix()[0]
            s.sueltas.append((lado, round(fr / fps, 2), round(N / fps, 2), round(float(v[fs:fs + N].max()), 2)))
            s.__dict__.setdefault('_ventanas_suelta', {}).setdefault(lado, []).append((fr / fps, N / fps))
        if not s.sueltas:
            c[id(clip)] = (clip, clip); return clip
        nuevo = copy.copy(clip)
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

    SUELTA_ALISA = 4.0         # cuadros: al final, el brazo entero (como queda en Max) se alisa en la salida del objeto

    def globales(s, clip, t, bucle=False, pies=True):
        s._t_suelta = float(t)
        G, Pm, b = super().globales(s._soltar(clip), t, bucle, pies)
        if not (s.SUELTA_SUAVE and s.SUELTA_ALISA > 0) or getattr(s, '_haciendo_suelta', False):
            return G, Pm, b
        plan = s._plan_suelta(clip, bucle)
        f = float(t) * 30.0
        for L, ventanas in plan.items():
            for f0, Q, masc in ventanas:
                i = f - f0
                if i < 0 or i > len(masc) - 1:
                    continue
                i0 = int(np.floor(i)); i1 = min(i0 + 1, len(masc) - 1); a_ = i - i0
                w = masc[i0] * (1 - a_) + masc[i1] * a_
                if w < 1e-3:
                    continue
                G, Pm = s._brazo_alisado(G, Pm, L, Q, i0, i1, a_, w)
        return G, Pm, b

    def _brazo_alisado(s, G, Pm, L, Q, i0, i1, a_, w):
        from scipy.spatial.transform import Rotation as Ro
        ix = s.idx
        cadena = [ix[L + n] for n in ('Arm', 'ForeArm', 'Hand')]
        G0, P0 = G.copy(), Pm.copy(); G = G.copy(); Pm = Pm.copy()
        for k_, j in enumerate(cadena):
            q = Q[i0, k_] * (1 - a_) + Q[i1, k_] * a_
            q /= np.linalg.norm(q)
            qa = Ro.from_matrix(G0[j]).as_quat()
            if qa @ q < 0: q = -q
            qm = qa * (1 - w) + q * w; qm /= np.linalg.norm(qm)
            G[j] = Ro.from_quat(qm).as_matrix()
        # el resto del brazo (huesos de giro, dedos) sigue a su padre como estaba; el hombro no se mueve
        for j in s._sub_de(cadena[0]):
            pa = s.padres[j]
            if j == cadena[0]:
                continue
            if j not in cadena:
                G[j] = G[pa] @ (G0[pa].T @ G0[j])
            Pm[j] = Pm[pa] + G[pa] @ (G0[pa].T @ (P0[j] - P0[pa]))
        return G, Pm

    def _plan_suelta(s, clip, bucle):
        c = s.__dict__.setdefault('_planes_suelta', {})
        if (id(clip), bucle) in c:
            return c[(id(clip), bucle)][1]
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation as Ro
        s._soltar(clip)
        vs = dict(getattr(s, '_ventanas_suelta', {}) or {})
        plan = {}
        s._haciendo_suelta = True
        try:
            for L, lista in vs.items():
                cadena = [s.idx[L + n] for n in ('Arm', 'ForeArm', 'Hand')]
                for a_, d_ in lista:
                    f0 = max(0, int(a_ * 30) - 12); f1 = int(np.ceil((a_ + d_) * 30)) + 14
                    Q = []
                    for f in range(f0, f1 + 1):
                        G, _, _ = s.globales(clip, f / 30.0, bucle, True)
                        Q.append([Ro.from_matrix(G[j]).as_quat() for j in cadena])
                    Q = np.array(Q)
                    for k_ in range(Q.shape[1]):
                        for f in range(1, len(Q)):
                            if Q[f, k_] @ Q[f - 1, k_] < 0: Q[f, k_] = -Q[f, k_]
                    Qs = gaussian_filter1d(Q, s.SUELTA_ALISA, axis=0, mode='nearest')
                    Qs /= np.linalg.norm(Qs, axis=2, keepdims=True)
                    # se usa desde que suelta hasta un poco después de terminar; entra y sale suave
                    ff = np.arange(f0, f1 + 1) / 30.0
                    u1 = np.clip((ff - a_) / 0.1, 0, 1); u2 = np.clip((a_ + d_ + 0.3 - ff) / 0.15, 0, 1)
                    m_ = np.minimum(u1, u2); m_ = m_ * m_ * (3 - 2 * m_)
                    plan.setdefault(L, []).append((f0, Qs, m_))
        finally:
            s._haciendo_suelta = False
        c[(id(clip), bucle)] = (clip, plan)
        return plan

    def _dedos(s, G, Pm, b):
        # lo que Max hace con la mano LIBRE (brazo relajado, brazo abierto...) entra al mismo paso suave que el brazo
        # sale del objeto (antes entraba de golpe en cuanto se abrían los dedos y daba un tirón a medio camino)
        r = super()._dedos(G, Pm, b)
        vs = getattr(s, '_ventanas_suelta', None); t = getattr(s, '_t_suelta', None)
        if s.SUELTA_SUAVE and s.SUELTA_LIBRE_SUAVE and vs and t is not None and hasattr(s, '_dedos_libres'):
            for L, lista in vs.items():
                for a_, d_ in lista:
                    if a_ <= t <= a_ + d_:
                        u = (t - a_) / d_; w = u ** 3 * (u * (u * 6 - 15) + 10)
                        s._dedos_libres[L] = min(s._dedos_libres.get(L, 1.0), w)
        return r


class _MangaLimpia:
    """BORDE DE LA MANGA (mejora 5): el borde se veía dentado (ondulado como papel cortado) y la capa de adentro de la
    camiseta asomaba como un segundo filo. Ahora, en cada cuadro: el borde de cada manga queda en un corte limpio (sin
    ondas a lo largo del brazo), la capa de adentro termina justo debajo de la de afuera (un solo borde) y el borde se
    redondea un poco hacia adentro, como un dobladillo con grosor. La tela de alrededor acompaña suave (sin escalón)."""
    MANGA_LIMPIA = True
    MANGA_GROSOR = 0.0022      # m entre la capa de afuera y la de adentro en el borde
    MANGA_REDONDEA = 0.0012    # m que el filo de afuera se mete hacia el brazo (borde redondeado)
    MANGA_ONDAS = 3            # ondas que se dejan en el contorno del borde (más = sigue más los bultos)
    MANGA_AFLOJA = 0.005       # m máximos que el borde se abre para no quedar más adentro que antes
    MANGA_BRILLO = 0.55        # cuánto se inclinan las normales del filo hacia el codo (se ve el grosor del dobladillo)

    def _lazos_manga(s, k):
        c = s.__dict__.setdefault('_lazos_manga_c', {})
        if k in c:
            return c[k]
        d = s._malla_tela(k); P0 = s.prims[k]['P'].astype(float)[d['rep']]
        out = []
        for lz in d['lazos']:
            cx = P0[lz].mean(0)
            dl = np.linalg.norm(cx - s.pA['LeftArm']); dr = np.linalg.norm(cx - s.pA['RightArm'])
            lado = 'Left' if dl < dr else 'Right'
            if min(dl, dr) < 0.25 and abs(cx[0]) > 0.12:          # solo las mangas (no el cuello ni el bajo)
                out.append((lado, lz))
        c[k] = out
        return out

    def _tela_sobre_piel(s, k, V, piel, margen):
        V = super()._tela_sobre_piel(k, V, piel, margen)
        if not s.MANGA_LIMPIA:
            return V
        nm = s.prims[k]['mname']
        if nm.startswith('Poloche') and 'Interior' not in nm:
            return s._manga_afuera(k, V)
        if nm.startswith('Poloche') and 'Interior' in nm:
            return s._manga_adentro(k, V)
        return V

    def _eje_brazo(s, L):
        A = s.Wp[s.idx[L + 'Arm']]; E = s.Wp[s.idx[L + 'ForeArm']]
        e = E - A; return A, e / (np.linalg.norm(e) + 1e-12)

    def _vecinos_siguen(s, d, X, D):
        """Los anillos de al lado del borde se mueven una parte de lo que se movió el borde (sin escalón)."""
        A = d['A']; mb = np.zeros(len(X), bool); mb[np.abs(D).sum(1) > 0] = True
        cnt = np.asarray(A @ mb.astype(float)).ravel()
        prom = (A @ D) / np.maximum(cnt, 1)[:, None]
        r1 = (cnt > 0) & ~mb
        X[r1] += 0.55 * prom[r1]
        D1 = np.zeros_like(D); D1[r1] = 0.55 * prom[r1]
        cnt2 = np.asarray(A @ r1.astype(float)).ravel()
        r2 = (cnt2 > 0) & ~mb & ~r1
        X[r2] += 0.45 * ((A @ D1) / np.maximum(cnt2, 1)[:, None])[r2]
        return X

    def _manga_afuera(s, k, V):
        d = s._malla_tela(k); X = V[d['rep']].copy(); D = np.zeros_like(X)
        s._borde_manga = {}
        for L, lz in s._lazos_manga(k):
            A, e = s._eje_brazo(L)
            Y = X[lz]; cY = Y.mean(0)
            # corte limpio: el borde queda en un plano (el que mejor le cuadra) -> sin ondas a lo largo del brazo
            _, _, Vt = np.linalg.svd(Y - cY); nrm = Vt[2]
            if nrm @ e < 0: nrm = -nrm
            Y = Y - np.outer((Y - cY) @ nrm, nrm)
            # dentro del plano, el contorno suave (sin dientes ni bultos): el radio se deja solo con sus ondas largas
            # (forma de óvalo) y se agranda lo justo para que ningún punto quede más adentro que antes (no asoma piel)
            cY = Y.mean(0); R_ = Y - cY; r = np.linalg.norm(R_, axis=1) + 1e-12
            F_ = np.fft.rfft(r); F_[s.MANGA_ONDAS + 1:] = 0; rl = np.fft.irfft(F_, len(r))
            rl = rl + min(max(float((r - rl).max()), 0.0), s.MANGA_AFLOJA)
            Y = cY + R_ * (rl / r)[:, None]
            rad = (Y - A) - np.outer((Y - A) @ e, e); rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-12
            Y = Y - rad * s.MANGA_REDONDEA
            D[lz] = Y - X[lz]
            s._borde_manga[L] = (Y.copy(), rad, nrm, cY, float(np.linalg.norm(Y - cY, axis=1).max()))
        if not s._borde_manga:
            return V
        Xn = X + D
        Xn = s._vecinos_siguen(d, Xn, D)
        return s._detras_del_borde(d, Xn, s._lazos_manga(k), 0.003)[d['inv']]

    def _detras_del_borde(s, d, X, lazos, atras):
        """Ningún punto de la manga (fuera del propio borde) pasa del borde: así el filo es el borde limpio y no un
        pliegue de la tela que sobresale (eso hacía los "dientes")."""
        en_borde = np.zeros(len(X), bool)
        for _, lz in lazos: en_borde[lz] = True
        for L, (Y, rad, nrm, cY, rmax) in s._borde_manga.items():
            u = (X - cY) @ nrm
            ra = np.linalg.norm((X - cY) - np.outer(u, nrm), axis=1)
            sel = (~en_borde) & (ra < rmax + 0.02) & (u > -0.03) & (u > -atras)
            if sel.any():
                X[sel] -= np.outer(u[sel] + atras, nrm)
        return X

    def _manga_adentro(s, k, V):
        bm = getattr(s, '_borde_manga', None)
        if not bm:
            return V
        d = s._malla_tela(k); X = V[d['rep']].copy(); D = np.zeros_like(X)
        c = s.__dict__.setdefault('_par_manga', {})
        if k not in c:                                   # cada punto del borde de adentro con su punto de afuera
            ka = next(i for i, p in enumerate(s.prims) if p['mname'].startswith('Poloche') and 'Interior' not in p['mname'])
            da = s._malla_tela(ka); Pa = s.prims[ka]['P'].astype(float)[da['rep']]
            Pi = s.prims[k]['P'].astype(float)[d['rep']]
            par = {}
            afuera = dict(s._lazos_manga(ka))
            for L, lz in s._lazos_manga(k):
                if L not in afuera:
                    continue
                lza = afuera[L]
                dd = np.linalg.norm(Pi[lz][:, None] - Pa[lza][None], axis=2)
                par[L] = (lz, np.argmin(dd, 1))
            c[k] = par
        for L, (lz, jj) in c[k].items():
            if L not in bm:
                continue
            Ya, rad = bm[L][0], bm[L][1]
            D[lz] = (Ya[jj] - rad[jj] * s.MANGA_GROSOR) - X[lz]
        Xn = s._vecinos_siguen(d, X + D, D)
        return s._detras_del_borde(d, Xn, s._lazos_manga(k), 0.004)[d['inv']]

    def pose_mats(s, G, Pm, b, apply=True):
        res = super().pose_mats(G, Pm, b, apply)
        bm = getattr(s, '_borde_manga', None)
        if not (s.MANGA_LIMPIA and bm and s.MANGA_BRILLO > 0):
            return res
        # normales del filo de afuera: un poco hacia el codo (se ve el canto redondeado del dobladillo)
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        for k, p in enumerate(s.prims):
            nm = p['mname']
            if not nm.startswith('Poloche') or 'Interior' in nm:
                continue
            d = s._malla_tela(k)
            for L, lz in s._lazos_manga(k):
                if L not in bm:
                    continue
                rad, nrm = bm[L][1], bm[L][2]
                w = np.zeros(len(d['rep'])); w[lz] = s.MANGA_BRILLO
                nr = np.zeros((len(d['rep']), 3)); nr[lz] = (rad + nrm) @ Cz.T
                wv = w[d['inv']][p['idx']]; sel = wv > 0
                if not sel.any():
                    continue
                V_, N_ = res[k]
                N_ = N_.copy(); tgt = nr[d['inv']][p['idx']][sel]
                tgt /= np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12
                N_[sel] = N_[sel] * (1 - wv[sel, None]) + tgt * wv[sel, None]
                N_[sel] /= np.linalg.norm(N_[sel], axis=1, keepdims=True) + 1e-12
                res[k] = (V_, N_)
        return res


class MaxV14(_MangaLimpia, _CodoSinSaltos, _CodoEstable, _SueltaSuave, _Erguido, _SinTirones, MaxV12):
    """Max V14 sin cara viva."""


class MaxCaraV14(_MangaLimpia, _CodoSinSaltos, _CodoEstable, _SueltaSuave, _Erguido, _SinTirones, MaxCaraV12):
    """La que se usa desde el ZIP 37: Max con cara viva y todas las mejoras."""
