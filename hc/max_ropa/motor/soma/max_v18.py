"""Max V18 = Max V17 (ZIP 43) + RODILLA QUE GIRA AL DOBLARSE + SENTADO BIEN APOYADO. Archivo NUEVO: no cambia max_v17.py.

Por qué (prueba del 3 nov 2026: video de Gemini Omni -> GEM-X -> Max, sentarse con las piernas cruzadas):
 - los pies no se cruzaban del todo: el tope de la rodilla (V16) no dejaba girar la pierna de abajo, pero una rodilla de
   verdad, MUY doblada, sí gira (~40°) y se ladea un poco: es lo que deja cruzar las piernas;
 - sentado quedaba flotando: no bajaba a apoyar las nalgas si un pie tocaba antes el piso.

 1. RODILLA QUE GIRA AL DOBLARSE (RODILLA_GIRA): con la rodilla doblada más de ~60° el tope de giro de la pierna de
    abajo sube de 30° a RODILLA_GIRO_DOBLADA (45°) y el de ladeo de 18° a RODILLA_LADO_DOBLADA (40°), poco a poco hasta
    ~110°. Con la pierna estirada todo sigue igual que V17.
 2. SENTADO BIEN APOYADO: sentado en el piso, Max baja hasta apoyar las nalgas aunque un pie toque antes; ese pie sube
    doblando la pierna (no se mete en el piso).

    from soma.max_v18 import MaxCaraV18, PistaCaraV18
    m = MaxCaraV18(glb='/home/claude/Personajes/Max/max_v5.glb')
Se apaga con MaxCaraV18.RODILLA_GIRA = False. Todo lo de V17 sigue igual (y se apaga igual).
"""
import numpy as np

from .max_v16 import _giro_y_doblez, _rehacer
from .max_v17 import MaxCaraV17, MaxV17, PistaCaraV17, _Piso, _suave


class PistaCaraV18(PistaCaraV17):
    """Igual que PistaCaraV17."""


class _RodillaDoblada:
    RODILLA_GIRA = True
    RODILLA_GIRO_DOBLADA = 45.0     # grados de giro de la pierna de abajo con la rodilla muy doblada
    RODILLA_LADO_DOBLADA = 40.0     # grados de ladeo con la rodilla muy doblada
    RODILLA_DESDE, RODILLA_HASTA = 60.0, 110.0

    def _limitar(s, clip):
        nuevo = super()._limitar(clip)
        if not (s.RODILLA_GIRA and getattr(s, 'LIMITES', True)):
            return nuevo
        c = s.__dict__.setdefault('_clips_rodilla', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation as Ro
        tab = s._tabla_limites(); L = nuevo.L.copy(); cambio = False
        for nom in ('LeftShin', 'RightShin'):
            if nom not in clip.idx or nom not in tab:
                continue
            t = tab[nom]; j = clip.idx[nom]; N, a, h = t['N'], t['a'], t['h']
            R = np.einsum('ji,njk->nik', N, clip.L[:, j])                 # la pierna como la pidió la animación
            tw, sw = _giro_y_doblez(R, a)
            twd = np.degrees(tw); hg = np.degrees(sw @ h); off = sw - np.outer(sw @ h, h)
            dob = gaussian_filter1d(_suave((np.abs(hg) - s.RODILLA_DESDE) / (s.RODILLA_HASTA - s.RODILLA_DESDE)), 2.0, mode='nearest')
            if dob.max() < 1e-3:
                continue
            gmax = s.GIRO_MAX['Shin'] * (1 - dob) + s.RODILLA_GIRO_DOBLADA * dob
            omax = s.BISAGRA['Shin'] * (1 - dob) + s.RODILLA_LADO_DOBLADA * dob
            hlo, hhi = t['hg'][0] - 5, t['hg'][1] + 5
            twc = np.clip(twd, -gmax, gmax); hgc = np.clip(hg, hlo, hhi)
            on = np.degrees(np.linalg.norm(off, axis=1)); k = np.minimum(1, omax / np.maximum(on, 1e-9))
            swc = np.outer(np.radians(hgc), h) + off * k[:, None]
            Rn = np.einsum('nij,njk->nik', Ro.from_rotvec(swc).as_matrix(), Ro.from_rotvec(np.outer(np.radians(twc), a)).as_matrix())
            Ln = np.einsum('ij,njk->nik', N, Rn)
            L[:, j] = s._mezclar(L[:, j], Ln, dob); cambio = True
        out = _rehacer(nuevo, L) if cambio else nuevo
        c[id(clip)] = (clip, out)
        return out


class _Piso18(_Piso):
    def _suelo(s, clip, t):
        """Altura a sumar a todo Max para que el pie más bajo pise el piso cuando la animación está en el piso (no en el
        aire): Max no tiene las mismas medidas que la persona de la animación y quedaba unos cm hundido o flotando."""
        c = s.__dict__.setdefault('_suelo_clip', {})
        if id(clip) not in c:
            from scipy.ndimage import gaussian_filter1d
            v = np.zeros(clip.n); w = np.zeros(clip.n)
            pc = np.minimum.reduce([clip.P[:, clip.idx[n], 1] for n in ('LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot')
                                    if n in clip.idx])
            alto = pc - np.percentile(pc, 3)                                   # pies sobre el piso de la animación
            for f in range(clip.n):
                G, Pm, b = super(_Piso, s).globales(clip, f / clip.fps)
                h, mano = s._alturas(G, Pm, b)
                v[f] = -h[mano['pies']].min()
                w[f] = _suave((s.SUELO_AIRE - alto[f]) / 0.04)
            c[id(clip)] = (clip, gaussian_filter1d(v * w, s.SUELO_SUAVE * clip.fps, mode='nearest'))
        v = c[id(clip)][1]
        return float(np.interp(t * clip.fps, np.arange(len(v)), v)) if s.SUELO_PIES else 0.0

    def _baja_sentado(s, clip, t):
        """Cuánto bajar a Max sentado en cada momento, calculado para toda la animación y suavizado (sin saltos)."""
        c = s.__dict__.setdefault('_bajas_sentado', {})
        if id(clip) not in c:
            from scipy.ndimage import gaussian_filter1d
            v = np.zeros(clip.n)
            for f in range(clip.n):
                G, Pm, b = super(_Piso, s).globales(clip, f / clip.fps)
                su = s._suelo(clip, f / clip.fps); Pm = Pm.copy(); Pm[:, 1] += su / s.base.scale
                h, mano = s._alturas(G, Pm, b)
                hc = h[mano['cuerpo']].min()
                if s.PISO_SOLIDO and hc < 0:
                    h = h - hc; Pm = Pm.copy(); Pm[:, 1] += -hc / s.base.scale
                cadera = s._altura_j(Pm, s.idx['Hips']); nal = h[mano['nalgas']].min()
                v[f] = _suave((s.SENTADO_CADERA - cadera) / 0.10) * _suave((s.SENTADO_HUECO_18 - nal) / 0.05) * max(0.0, nal)
            c[id(clip)] = (clip, gaussian_filter1d(v, s.SENTADO_SUAVE * clip.fps, mode='nearest'))
        v = c[id(clip)][1]
        return float(np.interp(t * clip.fps, np.arange(len(v)), v))

    """El piso de V17 con el sentado bien apoyado (reemplaza a _Piso de V17, no se suma)."""
    SENTADO_IK = False          # True: baja la cadera doblando más las piernas (probado 3 nov: sube las rodillas, se ve peor)
    SUELO_PIES = True           # el pie más bajo pisa el piso (cuando la animación no está en el aire)
    SUELO_AIRE = 0.10           # m: con los pies más alto que esto sobre el piso de la animación, está en el aire (salto)
    SUELO_SUAVE = 0.10          # s: suavizado
    SENTADO_SUAVE = 0.12        # s: suavizado de cuánto baja (entra y sale sin saltos)
    SENTADO_HUECO_18 = 0.30     # m: hasta este hueco entre nalgas y piso se baja a apoyarlas (V17: 0,15)

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super(_Piso, s).globales(clip, t, bucle, pies)      # salta el _Piso de V17 (este lo reemplaza)
        su = s._suelo(clip, t)
        if abs(su) > 1e-4:
            manos0 = {L: Pm[s.idx[L + 'Hand']].copy() for L in ('Left', 'Right')}
            G = G.copy(); Pm = Pm.copy(); Pm[:, 1] += su / s.base.scale
            # la mano que agarra o sostiene se queda en el objeto (el brazo se acomoda)
            if hasattr(s, '_peso_agarre'):
                for L, h0 in manos0.items():
                    w = s._peso_agarre(L, t)
                    if w > 1e-3:
                        s._ik_brazo(G, Pm, b, L, Pm[s.idx[L + 'Hand']] * (1 - w) + h0 * w)
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
            baja = s._baja_sentado(clip, t)
            if baja > 1e-4:
                # Max es más alto que la persona de la animación: sentado, su cadera quedaba alta. Baja la cadera y las
                # piernas se doblan más para que los pies se queden donde estaban (como al agacharse de verdad)
                if s.SENTADO_IK:
                    pies0 = {L: Pm[s.idx[L + 'Foot']].copy() for L in ('Left', 'Right')}
                    Pm[:, 1] -= baja / s.base.scale
                    for L, f0 in pies0.items():
                        s._ik_pierna(G, Pm, b, L, f0)
                    h = s._alturas(G, Pm, b)[0]
                else:
                    baja = min(baja, max(0.0, h[mano['pies']].min()))        # sin meter los pies en el piso
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



def _rot_entre(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12); v = np.cross(a, b); c = float(a @ b)
    if c < -0.9999:
        return -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class _PiesPlanos:
    """El pie apoyado queda plano en el piso. Las animaciones sacadas de video (GEM-X) a veces dejan la punta del pie
    muy inclinada hacia abajo y Max quedaba de puntillas. Con el pie quieto y cerca del piso, el pie y los dedos se
    enderezan a la inclinación de un pie parado normal (la de las animaciones de Kimodo)."""
    PIES_PLANOS = True
    PIE_INCLINA = 22.0      # grados: tobillo -> base de los dedos, parado normal
    DEDOS_INCLINA = 14.0    # grados: base de los dedos -> punta

    def _apoyo_pie(s, clip, L):
        c = s.__dict__.setdefault('_apoyo_pies', {})
        k = (id(clip), L)
        if k not in c:
            from scipy.ndimage import gaussian_filter1d
            p = clip.P[:, clip.idx[L + 'Foot']]
            v = np.r_[0, np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * clip.fps]
            hp = p[:, 1] - np.percentile(p[:, 1], 3)
            w = _suave((0.35 - gaussian_filter1d(v, 1.5)) / 0.15) * _suave((0.08 - hp) / 0.04)
            c[k] = gaussian_filter1d(w, 2.0, mode='nearest')
        return c[k]

    def _aplanar(s, G, Pm, b, L, w):
        ix = s.idx; iF, iT, iE = ix[L + 'Foot'], ix[L + 'ToeBase'], ix[L + 'ToeEnd']
        for (i0, i1, inc) in ((iF, iT, s.PIE_INCLINA), (iT, iE, s.DEDOS_INCLINA)):
            d = Pm[i1] - Pm[i0]; hz = np.array([d[0], 0.0, d[2]]); nh = np.linalg.norm(hz)
            if nh < 1e-6:
                continue
            hz /= nh; ang = np.radians(inc)
            des = hz * np.cos(ang) - np.array([0, 1.0, 0]) * np.sin(ang)
            act = d / np.linalg.norm(d)
            if act @ des > 0.99995:
                continue
            from scipy.spatial.transform import Rotation as Ro
            rv = Ro.from_matrix(_rot_entre(act, des)).as_rotvec() * w
            R = Ro.from_rotvec(rv).as_matrix()
            sub = [i0] + s._hijos_de(i0)
            for k in sub:
                G[k] = R @ G[k]
            for k in s._hijos_de(i0):
                Pm[k] = Pm[s.padres[k]] + G[s.padres[k]] @ b['off'][k]

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if not s.PIES_PLANOS:
            return G, Pm, b
        f = int(np.clip(round(t * clip.fps), 0, clip.n - 1)); G = G.copy(); Pm = Pm.copy(); cambio = False
        for L in ('Left', 'Right'):
            w = float(s._apoyo_pie(clip, L)[f])
            if w > 1e-3:
                s._aplanar(G, Pm, b, L, w); cambio = True
        if cambio:      # el pie más bajo vuelve a pisar el piso (el talón baja, no se hunde la punta)
            h, mano = s._alturas(G, Pm, b)
            hp = h[mano['pies']].min()
            if abs(hp) > 1e-4:
                manos0 = {L: Pm[s.idx[L + 'Hand']].copy() for L in ('Left', 'Right')}
                Pm[:, 1] -= hp / s.base.scale
                if hasattr(s, '_peso_agarre'):
                    for L, h0 in manos0.items():
                        wa = s._peso_agarre(L, t)
                        if wa > 1e-3:
                            s._ik_brazo(G, Pm, b, L, Pm[s.idx[L + 'Hand']] * (1 - wa) + h0 * wa)
        return G, Pm, b


class _ManosComoVideo:
    """Para animaciones sacadas de video con GEM-X: las manos que en el video tocan el cuerpo o la otra mano (aplaudir,
    manos en la cintura, rascarse la cabeza, brazos cruzados) llegan también en Max. Se usa la posición de las manos
    respecto al tronco de la persona del video, ajustada a las medidas de Max. Cargar con
        m.cargar_video(clip, '<carpeta de gemx_out>/movimiento.npz')"""
    MANOS_VIDEO = True
    MANO_CUERPO_CERCA = 0.30    # m: mano a menos de esto del tronco/cabeza en el video = tocando o casi
    MANO_TOCA = 0.23            # m: palma a menos de esto del hueso en el video = tocando (se apoya en la piel)
    MANO_PIEL = 0.03            # m: centro de la palma sobre la piel (grosor de la mano)
    PALMAS_CERCA = 0.30         # m: palmas a menos de esto en el video = se juntan
    _TRONCO = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'LeftLeg', 'RightLeg')

    def cargar_video(s, clip, npz):
        d = np.load(npz, allow_pickle=True)
        J = d['global_joints'][0]; nombres = [str(x) for x in d['public_joint_names']]
        if len(nombres) == J.shape[1] + 1:
            nombres = nombres[1:]
        s.__dict__.setdefault('_gem', {})[id(clip)] = (clip, J, {n: i for i, n in enumerate(nombres)})

    @staticmethod
    def _marco(P, g):
        o = P[g['Hips']]; up = P[g['Chest']] - o; lt = P[g['LeftArm']] - P[g['RightArm']]
        up = up / np.linalg.norm(up); lt = lt - up * (lt @ up); lt = lt / np.linalg.norm(lt); fw = np.cross(lt, up)
        return o, np.stack([lt, up, fw])

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        e = s.__dict__.get('_gem', {}).get(id(clip))
        if not (s.MANOS_VIDEO and e):
            return G, Pm, b
        _, J, g = e
        x = np.clip(t * clip.fps, 0, len(J) - 1); i0 = int(x); i1 = min(i0 + 1, len(J) - 1); a = x - i0
        P = J[i0] * (1 - a) + J[i1] * a
        ix = s.idx; M = {n: Pm[ix[n]] for n in ('Hips', 'Chest', 'LeftArm', 'RightArm')}
        og, Ag = s._marco(P, g); om, Am = s._marco(np.stack([M['Hips'], M['Chest'], M['LeftArm'], M['RightArm']]),
                                               {'Hips': 0, 'Chest': 1, 'LeftArm': 2, 'RightArm': 3})
        # medidas: ancho de hombros (de lado), largo del tronco (arriba), y el promedio (adelante)
        sl = np.linalg.norm(M['LeftArm'] - M['RightArm']) / np.linalg.norm(P[g['LeftArm']] - P[g['RightArm']])
        su = np.linalg.norm(M['Chest'] - M['Hips']) / np.linalg.norm(P[g['Chest']] - P[g['Hips']])
        esc = np.array([sl, su, (sl + su) / 2])
        palma_g = {L: (P[g[L + 'Hand']] + P[g[L + 'HandMiddle1']]) / 2 for L in ('Left', 'Right')}
        palma_m = {L: (Pm[ix[L + 'Hand']] + Pm[ix[L + 'HandMiddle1']]) / 2 for L in ('Left', 'Right')}
        dpal = np.linalg.norm(palma_g['Left'] - palma_g['Right'])
        w_pal = _suave((s.PALMAS_CERCA - dpal) / 0.12)
        objetivo = {}
        for L in ('Left', 'Right'):
            dists = {n: np.linalg.norm(palma_g[L] - P[g[n]]) for n in s._TRONCO}
            ancla = min(dists, key=dists.get); dcu = dists[ancla]
            w_cu = _suave((s.MANO_CUERPO_CERCA - dcu) / 0.12)
            # con el brazo colgando estirado no hay contacto (la mano pasa cerca de la cadera sin tocarla)
            u1 = P[g[L + 'ForeArm']] - P[g[L + 'Arm']]; u2 = P[g[L + 'Hand']] - P[g[L + 'ForeArm']]
            codo = np.degrees(np.arccos(np.clip(u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-9), -1, 1)))
            w_cu *= _suave((codo - 35) / 25)
            w = max(w_cu, w_pal)
            if hasattr(s, '_peso_agarre'):
                w *= 1.0 - s._peso_agarre(L, t)
            if w < 1e-3:
                continue
            # la palma respecto a la parte del cuerpo más cercana (cabeza, pecho, cadera...), girada al tronco de Max
            loc = Ag @ (palma_g[L] - P[g[ancla]])
            obj = Pm[ix[ancla]] + Am.T @ loc
            # tocando de verdad en el video (palma a menos de ~15 cm del hueso): la palma se apoya en la piel de Max
            toca = _suave((s.MANO_TOCA - dcu) / 0.05) * (1 - w_pal)
            if toca > 1e-3:
                J_, loc_, _ = s._muestras(b)
                xs = np.einsum('vij,vj->vi', G[J_], loc_) + Pm[J_]
                propio = np.isin(J_, [j for j, n in enumerate(s.nombres) if n.startswith(L) and ('Arm' in n or 'Hand' in n)])
                dd = np.linalg.norm(xs - obj, axis=1); dd[propio] = 9
                k = int(np.argmin(dd))
                if dd[k] > 0.01:
                    sup = xs[k] + (obj - xs[k]) / dd[k] * s.MANO_PIEL
                    obj = obj * (1 - toca) + sup * toca
            objetivo[L] = (obj, w)
        if w_pal > 1e-3 and len(objetivo) == 2:                    # al juntar las palmas: mismo hueco que en el video
            mid = (objetivo['Left'][0] + objetivo['Right'][0]) / 2
            dv = palma_g['Left'] - palma_g['Right']
            for L, sg in (('Left', 0.5), ('Right', -0.5)):
                o, w = objetivo[L]
                objetivo[L] = (o * (1 - w_pal) + (mid + sg * dv) * w_pal, w)
        if objetivo:
            G = G.copy(); Pm = Pm.copy()
            for L, (o, w) in objetivo.items():
                mov = (o - palma_m[L]) * w
                s._ik_brazo(G, Pm, b, L, Pm[ix[L + 'Hand']] + mov)
        return G, Pm, b


class MaxV18(_ManosComoVideo, _PiesPlanos, _Piso18, _RodillaDoblada, MaxV17):
    """Max V18 sin cara viva."""


class MaxCaraV18(_ManosComoVideo, _PiesPlanos, _Piso18, _RodillaDoblada, MaxCaraV17):
    """La que se usa desde el ZIP 44: Max V17 + rodilla que gira al doblarse + sentado bien apoyado."""
