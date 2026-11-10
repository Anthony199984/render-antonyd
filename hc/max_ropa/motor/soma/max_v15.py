"""ZIP 40 · Max V15 = Max V14 (ZIP 37) + CARA EN MOVIMIENTO. Archivo NUEVO: no cambia max_v14.py.

La cara de Max ya reaccionaba a lo que le ponía la escena (expresión por acción, esfuerzo al cargar algo pesado), pero
al trotar o correr quedaba neutra y con la mirada fija, como si estuviera parado. Ahora la cara reacciona SOLA a lo que
hace el cuerpo (no hay que marcarlo en la escena):
  - TROTE / CARRERA: respiración agitada al ritmo de los pasos (la boca se abre y se cierra un poco, sopla al soltar el
    aire, las mejillas se inflan apenas), cejas algo juntas y ojos un poco entrecerrados por el esfuerzo. Más fuerte
    cuanto más rápido va (empieza al trotar suave, ~1,2 m/s; del todo a ~2,6 m/s). Caminando no cambia.
  - MIRADA VIVA: los ojos hacen movimientos chiquitos (2-4°) cada 0,7-2 s, como una persona de verdad; nunca quedan
    clavados. Se suma a donde ya mira (mirada de la escena).
Se suma a la expresión de la escena (si la escena pone "sonrisa" mientras trota, sonríe y respira agitado).

    from soma.max_v15 import MaxCaraV15, PistaCaraV15
    m = MaxCaraV15(glb='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV15.desde_escena('<carpeta de salida de escena5.py o escena6.py>', '<escena.json>')
    res = m.pose(clip, t)
CAMISETA CON VIDA: el bajo y el borde de las mangas rebotan y se quedan un poco atrás al trotar; pliegues que corren por
el bajo (ver _CamisetaViva).
Se apaga con MaxCaraV15.CARA_TROTE = False / MIRADA_VIVA = False / CAMISETA_VIVA = False.
"""
import numpy as np

from .max_v14 import MaxCaraV14, MaxV14, PistaCaraV14


class PistaCaraV15(PistaCaraV14):
    """Igual que PistaCaraV14 (los agarres de la escena)."""


def _suave(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


class _CaraEnMovimiento:
    CARA_TROTE = True
    TROTE_DESDE = 1.2          # m/s de la cadera: desde aquí empieza la cara de trote
    TROTE_LLENO = 2.6          # m/s: del todo (trote normal ~2 m/s = 60 %)
    RESPIRA_PASOS = 2.0        # pasos por respiración (al trotar se respira cada 2 pasos)
    MIRADA_VIVA = True
    MIRADA_GRADOS = (2.0, 4.0)  # tamaño de los movimientos chiquitos de los ojos
    MIRADA_CADA = (0.7, 2.0)   # segundos entre uno y otro

    def _movimiento(s, clip):
        """Por cuadro del clip: velocidad de la cadera (m/s, alisada) y fase de la respiración (ligada a los pasos)."""
        c = s.__dict__.setdefault('_movimiento_c', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.ndimage import gaussian_filter1d
        ix = clip.idx; fps = clip.fps
        h = clip.P[:, ix['Hips']]
        v = np.r_[0.0, np.linalg.norm(np.diff(h[:, [0, 2]], axis=0), axis=1) * fps]
        v = gaussian_filter1d(v, 0.3 * fps, mode='nearest')
        # pasos: cada vez que un pie llega a su punto más bajo
        pasos = np.zeros(clip.n)
        for L in ('Left', 'Right'):
            if L + 'Foot' not in ix:
                continue
            y = gaussian_filter1d(clip.P[:, ix[L + 'Foot'], 1], 1.0)
            mn = np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
            pasos[mn] = 1
        cuenta = np.cumsum(pasos)
        # fase continua: entre paso y paso se interpola
        idx = np.where(pasos > 0)[0]
        fase = np.zeros(clip.n)
        if len(idx) >= 2:
            fase = np.interp(np.arange(clip.n), idx, np.arange(len(idx)))
        else:
            fase = np.arange(clip.n) / fps * 2.6
        fase = 2 * np.pi * fase / s.RESPIRA_PASOS
        c[id(clip)] = (clip, (v, fase))
        return c[id(clip)][1]

    def _saltos_mirada(s):
        if getattr(s, '_mirada', None) is None:
            rng = np.random.default_rng(11)
            t, lst = 0.0, []
            while t < 600:
                a = rng.uniform(0, 2 * np.pi); g = rng.uniform(*s.MIRADA_GRADOS)
                lst.append((t, g * np.cos(a), 0.6 * g * np.sin(a)))
                t += rng.uniform(*s.MIRADA_CADA)
            s._mirada = np.array(lst)
        return s._mirada

    def pesos_cara(s, G, t, b=None):
        w = super().pesos_cara(G, t, b)
        clip = getattr(s, '_clip_obj', None)
        if clip is not None and s.CARA_TROTE and 'Hips' in clip.idx:
            v, fase = s._movimiento(clip)
            f = float(np.clip(t * clip.fps, 0, clip.n - 1)); i0 = int(f); i1 = min(i0 + 1, clip.n - 1); u = f - i0
            vel = v[i0] * (1 - u) + v[i1] * u; ph = fase[i0] * (1 - u) + fase[i1] * u
            a = _suave((vel - s.TROTE_DESDE) / (s.TROTE_LLENO - s.TROTE_DESDE))
            s.cara_trote = a
            if a > 1e-3:
                sale = 0.5 + 0.5 * np.sin(ph)                  # 1 = soltando el aire
                add = {'jawOpen': 0.12 + 0.24 * sale, 'mouthFunnel': 0.22 * sale, 'cheekPuff': 0.14 * sale, 'mouthStretchLeft': 0.20, 'mouthStretchRight': 0.20,
                       'browDownLeft': 0.40, 'browDownRight': 0.40, 'browInnerUp': 0.25,
                       'eyeSquintLeft': 0.32, 'eyeSquintRight': 0.32, 'noseSneerLeft': 0.18 * (1 - sale),
                       'noseSneerRight': 0.18 * (1 - sale)}
                for k, val in add.items():
                    w[k] = float(np.clip(max(w.get(k, 0.0), 0.0) + val * a * (1.0 - 0.5 * w.get(k, 0.0)), 0, 1))
        if s.MIRADA_VIVA:
            M = s._saltos_mirada()
            i = int(np.searchsorted(M[:, 0], t) - 1)
            if i >= 0:
                # paso rápido (40 ms) del punto anterior al nuevo
                k = _suave((t - M[i, 0]) / 0.04)
                yaw = M[i - 1, 1] * (1 - k) + M[i, 1] * k if i > 0 else M[i, 1] * k
                pitch = M[i - 1, 2] * (1 - k) + M[i, 2] * k if i > 0 else M[i, 2] * k
                from .cara import OJOS_LADO, OJOS_ARRIBA
                for lado in ('Left', 'Right'):
                    if yaw >= 0:     # hacia la izquierda de Max: ojo izquierdo afuera, derecho adentro
                        kk = 'eyeLookOut' + lado if lado == 'Left' else 'eyeLookIn' + lado
                    else:
                        kk = 'eyeLookIn' + lado if lado == 'Left' else 'eyeLookOut' + lado
                    w[kk] = float(np.clip(w.get(kk, 0) + abs(yaw) / OJOS_LADO, 0, 1))
                    kp = ('eyeLookUp' if pitch >= 0 else 'eyeLookDown') + lado
                    w[kp] = float(np.clip(w.get(kp, 0) + abs(pitch) / OJOS_ARRIBA, 0, 1))
        return w


class _CamisetaViva:
    """CAMISETA CON VIDA (mejora 2): la tela suelta (el bajo de la camiseta y el borde de las mangas) tiene un poco de
    inercia: al trotar o moverse rápido rebota y se queda un pelo atrás del cuerpo, como tela gruesa de verdad (un
    resorte con freno, ~4 Hz), y en el bajo corren unos pliegues suaves con el movimiento. Después la tela pasa por lo
    de siempre (nunca atraviesa la piel, borde de manga limpio). Quieto no cambia nada."""
    CAMISETA_VIVA = True
    TELA_HZ = 4.0              # lo dura que es la tela (más = rebota más rápido y menos)
    TELA_FRENO = 0.28          # amortiguación (más = rebota menos veces)
    TELA_MAX = 0.028           # m: lo más que se separa la tela de donde iría pegada al cuerpo
    TELA_BAJO = 0.20           # m: alto de la parte de abajo de la camiseta que se mueve suelta
    TELA_MANGA = 0.10          # m: largo del borde de la manga que se mueve suelto
    PLIEGUE_VIVO_MM = 3.0      # alto de los pliegues que corren por el bajo

    def _plan_tela(s, clip):
        c = s.__dict__.setdefault('_plan_tela_c', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.ndimage import gaussian_filter1d
        ix = clip.idx; fps = clip.fps; n = clip.n
        fuentes = {'torso': clip.P[:, ix['Chest']]}
        for L in ('Left', 'Right'):
            if L + 'Arm' in ix and L + 'ForeArm' in ix:
                fuentes[L] = 0.6 * clip.P[:, ix[L + 'Arm']] + 0.4 * clip.P[:, ix[L + 'ForeArm']]
        w = 2 * np.pi * s.TELA_HZ; z = s.TELA_FRENO
        plan = {}
        for k, X in fuentes.items():
            X = gaussian_filter1d(X, 1.0, axis=0, mode='nearest')
            A = np.gradient(np.gradient(X, axis=0), axis=0) * fps * fps
            sub = 8; dt = 1.0 / (fps * sub)
            x = np.zeros(3); v = np.zeros(3); out = np.zeros((n, 3))
            for f in range(n):
                for q in range(sub):
                    a = A[f] if f == n - 1 else A[f] + (A[f + 1] - A[f]) * q / sub
                    acc = -w * w * x - 2 * z * w * v - a
                    v = v + acc * dt; x = x + v * dt
                out[f] = x
            nr = np.linalg.norm(out, axis=1, keepdims=True)
            out = out * np.minimum(1, s.TELA_MAX / np.maximum(nr, 1e-9))
            plan[k] = out
        c[id(clip)] = (clip, plan)
        return plan

    def _pesos_tela(s, k):
        c = s.__dict__.setdefault('_pesos_tela_c', {})
        if k in c:
            return c[k]
        d = s._malla_tela(k); q = s._prep_axila(k)
        P0 = s.prims[k]['P'].astype(float)[d['rep']]
        manga = np.maximum(q['manga']['Left'], q['manga']['Right'])
        cuerpo = manga < 0.5
        ymin = P0[cuerpo, 1].min() if cuerpo.any() else P0[:, 1].min()
        bajo = np.clip(1 - (P0[:, 1] - ymin) / s.TELA_BAJO, 0, 1) ** 1.5 * (1 - manga)
        mangas = {}
        for L in ('Left', 'Right'):
            lz = [lz for L_, lz in (s._lazos_manga(k) if hasattr(s, '_lazos_manga') else []) if L_ == L]
            if lz:
                cen = P0[lz[0]].mean(0)
                dist = np.linalg.norm(P0 - cen, axis=1)
                mangas[L] = np.clip(1 - dist / s.TELA_MANGA, 0, 1) ** 1.2 * q['manga'][L]
            else:
                mangas[L] = np.zeros(len(P0))
        # ángulo alrededor del cuerpo (para los pliegues que corren por el bajo)
        cx = P0[cuerpo].mean(0) if cuerpo.any() else P0.mean(0)
        ang = np.arctan2(P0[:, 2] - cx[2], P0[:, 0] - cx[0])
        c[k] = (bajo, mangas, ang)
        return c[k]

    def _axila_lisa(s, k, V):
        V = super()._axila_lisa(k, V)
        clip = getattr(s, '_clip_obj', None)
        if not s.CAMISETA_VIVA or clip is None or 'Chest' not in clip.idx:
            return V
        plan = s._plan_tela(clip)
        t = getattr(s, '_t', 0.0)
        f = float(np.clip(t * clip.fps, 0, clip.n - 1)); i0 = int(f); i1 = min(i0 + 1, clip.n - 1); u = f - i0
        en = lambda a: a[i0] * (1 - u) + a[i1] * u
        d = s._malla_tela(k)
        bajo, mangas, ang = s._pesos_tela(k)
        X = V[d['rep']]
        D = np.outer(bajo, en(plan['torso']))
        for L in ('Left', 'Right'):
            if L in plan:
                D += np.outer(mangas[L], en(plan[L]))
        # pliegues que corren por el bajo, más marcados cuanto más se mueve la tela
        act = float(np.clip(np.linalg.norm(en(plan['torso'])) / s.TELA_MAX, 0, 1))
        if act > 0.02 and s.PLIEGUE_VIVO_MM > 0:
            Wp = getattr(s, 'Wp', None)
            if Wp is not None:
                cen = 0.5 * (Wp[s.idx['Hips']] + Wp[s.idx['Chest']])
                rad = X - cen; rad[:, 1] = 0; rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
                ola = np.sin(4 * ang + 2 * np.pi * 1.6 * t)
                D += rad * (s.PLIEGUE_VIVO_MM / 1000 * act * bajo * 0.5 * (1 + ola))[:, None]
        s.tela_viva_mm = float(np.abs(D).max() * 1000) if len(D) else 0.0
        return (X + D)[d['inv']]


class MaxV15(MaxV14):
    """Max V15 sin cara viva (igual que V14: la cara en movimiento necesita la cara viva)."""


class MaxCaraV15(_CaraEnMovimiento, _CamisetaViva, MaxCaraV14):
    """La que se usa desde el ZIP 40: Max V14 + cara en movimiento (trote y mirada viva) + camiseta con vida."""
