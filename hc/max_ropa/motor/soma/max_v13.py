"""ZIP 36 · Max V13 = Max V12 (ZIP 34) + SIN TIRONES + BRAZO ABIERTO AL AGARRAR + CODO ESTABLE + CODO SIN SALTOS.

Kimodo / Motion Bricks a veces dejan un TIRÓN de 1-2 cuadros en un brazo (sobre todo donde se une una transición IA
con la animación siguiente): el brazo salta 11-12° en un cuadro y el codo gira hasta 45°, y al cuadro siguiente
vuelve. Max V13 busca esos tirones en la animación ANTES de todo lo demás y los alisa SOLO ahí (unos cuadros antes y
después); el resto queda como estaba (las manos que sostienen un objeto y los pies, exactamente igual). Un movimiento rápido de verdad (un golpe, un saludo) no se
toca, porque dura varios cuadros: solo cuenta como tirón un cuadro mucho más rápido que sus vecinos. Tampoco se toca
un brazo mientras su mano AGARRA algo (el objeto va pegado a la mano tal como lo dejó la escena).

Uso (igual que Max V12, cambia solo la clase):
    from soma.max_v13 import MaxCaraV13, PistaCaraV13
    m = MaxCaraV13(glb='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV13.desde_escena('<carpeta de salida de escena5.py>', '<escena.json>')
    res = m.pose(clip, t)

Se apaga con MaxCaraV13.SIN_TIRONES = False. Lo que se alisó queda en m.tirones (lista de (cuadro, grupo, grados)).
CODO ESTABLE (MaxCaraV13.CODO_ESTABLE): la causa de los giros de 100-150° del codo al alcanzar algo con el brazo casi
estirado; ahí el codo va hacia su lado natural (abajo y un poco afuera). Y CODO NATURAL AL AGARRAR
(CODO_NATURAL_AGARRE): agarres_ik (escena5.py) pone el codo unos 50° hacia afuera mientras la mano va al objeto y lo
sostiene; el brazo terminaba metido hacia el cuerpo o detrás del torso. Max ABRE el brazo: el codo va hacia afuera
(separado del cuerpo) y un poco hacia abajo, con la mano en el mismo sitio. Ver _CodoEstable.
Además CODO SIN SALTOS (MaxCaraV13.CODO_SIN_SALTOS): donde el codo todavía girara de golpe alrededor de la línea hombro-mano
(30-90° en un cuadro, al acomodar el brazo para un agarre con el brazo casi estirado) el giro se reparte en unos
cuadros; la mano no se mueve. Los saltos arreglados quedan en m.codo_saltos.
"""
import numpy as np

from .max_v12 import MaxCaraV12, MaxV12, PistaCaraV12

class PistaCaraV13(PistaCaraV12):
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
            prev = {}
            for f in range(n):
                G, Pm, b = super(_CodoSinSaltos, s).globales(cl, f / 30.0, bucle, True)
                for L in ('Left', 'Right'):
                    iA, iF, iH = s._brazo_ids(L)
                    A, E, H = Pm[iA], Pm[iF], Pm[iH]
                    e = (H - A) / (np.linalg.norm(H - A) + 1e-12)
                    p = (E - A) - e * ((E - A) @ e)
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
                plan[L] = ((xs - x) * masc + (xa - x) * (1 - masc), saltos)
            return plan

        plan = s._con_disco('codo13', clip, calcular) if hasattr(s, '_con_disco') else calcular(clip)
        c[(id(clip), bucle)] = (clip, plan)
        return plan

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if not s.CODO_SIN_SALTOS:
            return G, Pm, b
        plan = s._plan_codo13(clip, bucle)
        s.codo_saltos = {L: plan[L][1] for L in plan}
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


class MaxV13(_CodoSinSaltos, _CodoEstable, _SinTirones, MaxV12):
    """Max V12 (sin cara viva) + sin tirones + codo sin saltos."""


class MaxCaraV13(_CodoSinSaltos, _CodoEstable, _SinTirones, MaxCaraV12):
    """La que se usa desde el ZIP 36: Max V12 con cara viva + sin tirones + codo sin saltos."""
