"""ZIP 43 · Max V16 = Max V15 (ZIP 40) + LÍMITES DEL CUERPO + MIRAR ARRIBA REPARTIDO + MOVIMIENTO MENOS ROBÓTICO.
Archivo NUEVO: no cambia max_v15.py.

Por qué: las animaciones de Kimodo a veces piden lo que un cuerpo no puede. Medido en 441 animaciones: la cabeza de
"pt2_mira_avion_a" (mirar un avión) gira 176° sobre su eje (media vuelta: se veía dislocada), un antebrazo gira 160°,
rodillas giradas 170-180° en caídas, pecho 119°. Además, al mirar arriba casi todo el giro se lo llevaba el cuello.

 1. LÍMITES DEL CUERPO (LIMITES): cada articulación tiene su tope de GIRO sobre su propio hueso (no se retuerce) y de
    DOBLEZ. El codo y la rodilla son bisagras: solo se doblan hacia su lado y no se doblan al revés. Los dedos no se
    retuercen, no se doblan hacia atrás ni de lado. Los topes salen de esas 441 animaciones (lo normal) y del cuerpo
    humano; cuando una animación pide más, el movimiento se frena SUAVE (sin salto) en el tope. Ver soma/limites_soma77.npz.
 2. MIRAR ARRIBA REPARTIDO (MIRADA_REPARTIDA): como una persona de verdad: el cuello llega hasta ~30° hacia atrás; lo
    que pase de ahí se reparte: la mitad el pecho (se echa un poco hacia atrás, hasta 20°) y la otra mitad los ojos
    (miran más arriba, hasta 20°). La cara sigue mirando a donde miraba.
 2b. CUELLO A LOS LADOS (CUELLO_TOTAL): sumando cuello y cabeza, la cabeza gira a un lado como máximo ~70° respecto al
    pecho (nunca mirando hacia atrás ni con la barbilla encima del hombro) y se ladea como máximo ~35° (oreja hacia el
    hombro). Lo que la animación pida de más lo reparten el pecho (gira hasta 15° hacia ese lado) y los ojos (miran más
    de lado, hasta 22°). En "pt2_mira_avion_a" Kimodo daba la vuelta entera con la cabeza (363°).
 3. MENOS ROBÓTICO (SEGUIMIENTO): las partes no llegan todas a la vez: la cabeza sigue al pecho con un pequeño retraso
    y los brazos libres arrastran un poco detrás del hombro. Nunca en la mano que agarra o sostiene, ni en las piernas.

    from soma.max_v16 import MaxCaraV16, PistaCaraV16
    m = MaxCaraV16(glb='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV16.desde_escena('<carpeta de salida de escena5.py o escena6.py>', '<escena.json>')
Se apagan con MaxCaraV16.LIMITES / CUELLO_TOTAL / MIRADA_REPARTIDA / SEGUIMIENTO = False. Lo que se frenó queda en m.limitado
({articulación: grados de más}) m.cuello_frenado (grados de más a un lado y ladeando) y
m.mirada_repartida (grados).
"""
import copy
import os

import numpy as np

from .max_v15 import MaxCaraV15, MaxV15, PistaCaraV15

AQUI = os.path.dirname(os.path.abspath(__file__))


class PistaCaraV16(PistaCaraV15):
    """Igual que PistaCaraV15."""


def _suave(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def _rehacer(clip, L):
    """Clip nuevo con estas rotaciones locales (posiciones y globales recalculadas)."""
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
    return nuevo


def _rot(e, g):
    e = np.asarray(e, float); e = e / (np.linalg.norm(e) + 1e-12)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return np.eye(3) + np.sin(g) * K + (1 - np.cos(g)) * K @ K


def _giro_y_doblez(R, a):
    """R (n,3,3) = doblez @ giro(a). Devuelve giro (rad, -pi..pi) y el doblez como vector de rotación (n,3)."""
    from scipy.spatial.transform import Rotation as Ro
    q = Ro.from_matrix(R).as_quat()                       # x, y, z, w
    p = q[:, :3] @ a
    tw = 2 * np.arctan2(p, q[:, 3]); tw = (tw + np.pi) % (2 * np.pi) - np.pi
    Rt = Ro.from_rotvec(np.outer(tw, a)).as_matrix()
    S = np.einsum('nij,nkj->nik', R, Rt)                  # R @ Rt^T
    return tw, Ro.from_matrix(S).as_rotvec()


class _Limites:
    LIMITES = True
    # topes del cuerpo (grados): GIRO sobre el hueso y DOBLEZ (cono); el más chico entre esto y lo normal de los datos
    GIRO_MAX = {'Spine1': 15, 'Spine2': 15, 'Chest': 20, 'Neck1': 12, 'Neck2': 30, 'Head': 40, 'Shoulder': 25,
                'Arm': 100, 'ForeArm': 20, 'Hand': 95, 'Leg': 75, 'Shin': 30, 'Foot': 35, 'ToeBase': 8}
    DOBLEZ_MAX = {'Spine1': 45, 'Spine2': 30, 'Chest': 40, 'Neck1': 65, 'Neck2': 25, 'Head': 45, 'Shoulder': 45,
                  'Arm': 125, 'Hand': 85, 'Leg': 120, 'Foot': 62, 'ToeBase': 60}
    BISAGRA = {'ForeArm': 12, 'Shin': 18}       # codo y rodilla: lo más que se pueden torcer de lado
    MARGEN = 8                                  # grados más allá de lo normal de los datos antes de frenar
    # dedos: giro, doblez hacia la palma (mín..máx; negativo = hacia atrás) y de lado
    DEDO_GIRO = 10; DEDO_DOBLEZ = (-15, 100); DEDO_LADO = (15, 5)       # (nudillo, otras falanges)
    PULGAR_GIRO = 25; PULGAR_DOBLEZ = (-20, 85); PULGAR_LADO = 30

    def _tabla_limites(s):
        if getattr(s, '_tlim', None) is None:
            d = np.load(os.path.join(AQUI, 'limites_soma77.npz'))
            s._tlim = {str(n): dict(N=d['N'][i], a=d['a'][i], h=d['h'][i], tw=d['tw'][i], sw=float(d['sw'][i]),
                                     hg=d['hg'][i]) for i, n in enumerate(d['nombres'])}
            s._ejes = {str(n): d['ejes_todos'][i] for i, n in enumerate(d['ejes_todos_nombres'])}
        return s._tlim

    @staticmethod
    def _parte(n):
        for k in ('Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Shoulder', 'ForeArm', 'Arm', 'Hand', 'Leg',
                  'Shin', 'Foot', 'ToeBase'):
            if n == k or n == 'Left' + k or n == 'Right' + k:
                return k
        return None

    def _limitar(s, clip):
        if not s.LIMITES:
            return clip
        c = s.__dict__.setdefault('_clips_limitados', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        from scipy.spatial.transform import Rotation as Ro
        tab = s._tabla_limites()
        L = clip.L.copy(); n = clip.n
        s.limitado = {}
        # el brazo y los dedos de la mano que va a agarrar o sostiene quedan como los dejó el sistema de agarres
        tt = np.arange(n) / clip.fps
        libre = {lado: np.array([1.0 - s._peso_agarre(lado, x) for x in tt]) if hasattr(s, '_peso_agarre') else np.ones(n)
                 for lado in ('Left', 'Right')}
        BRAZO = ('Shoulder', 'Arm', 'ForeArm', 'Hand')

        def suave_donde(x_c, x_raw, satur):
            if not satur.any():
                return x_c
            m = maximum_filter1d(satur.astype(float), 7)
            m = gaussian_filter1d(m, 2.0)
            return x_c * (1 - m) + gaussian_filter1d(x_c, 3.0, mode='nearest') * m

        # ---- cuerpo
        for nom, t in tab.items():
            if nom not in clip.idx:
                continue
            parte = s._parte(nom)
            if parte is None:
                continue
            j = clip.idx[nom]; N, a = t['N'], t['a']
            R = np.einsum('ji,njk->nik', N, L[:, j])
            tw, sw = _giro_y_doblez(R, a)
            twd = np.degrees(tw)
            cap = s.GIRO_MAX[parte]
            lo = max(t['tw'][0] - s.MARGEN, -cap); hi = min(t['tw'][1] + s.MARGEN, cap)
            lo, hi = min(lo, -3.0), max(hi, 3.0)
            twc = np.clip(twd, lo, hi)
            twc = suave_donde(twc, twd, (twd < lo) | (twd > hi))
            if parte in s.BISAGRA:
                h = t['h']
                hg = np.degrees(sw @ h); off = sw - np.outer(sw @ h, h)
                hlo, hhi = t['hg'][0] - 5, t['hg'][1] + 5
                hgc = np.clip(hg, hlo, hhi); hgc = suave_donde(hgc, hg, (hg < hlo) | (hg > hhi))
                on = np.degrees(np.linalg.norm(off, axis=1)); omax = s.BISAGRA[parte]
                k = np.minimum(1, omax / np.maximum(on, 1e-9))
                k = suave_donde(k, k, on > omax)
                swc = np.outer(np.radians(hgc), h) + off * k[:, None]
                exceso = np.maximum.reduce([np.abs(twd - twc), np.abs(hg - hgc), np.maximum(on - omax, 0)])
            else:
                # con el giro MUY pasado (media vuelta) la inclinación que traía también viene al revés: se reduce
                # (si no, la cabeza quedaba agachada mirando al hombro)
                pasa = np.abs(twd - twc)
                red = np.clip(1 - (pasa - 45) / 90, 0.15, 1)
                red = gaussian_filter1d(red, 3.0, mode='nearest')
                sw = sw * red[:, None]
                mag = np.degrees(np.linalg.norm(sw, axis=1))
                smax = min(t['sw'] + s.MARGEN, s.DOBLEZ_MAX[parte])
                k = np.minimum(1, smax / np.maximum(mag, 1e-9))
                k = suave_donde(k, k, mag > smax)
                swc = sw * k[:, None]
                exceso = np.maximum(np.abs(twd - twc), np.maximum(mag - smax, 0))
            if exceso.max() < 0.5:
                continue
            Rn = np.einsum('nij,njk->nik', Ro.from_rotvec(swc).as_matrix(), Ro.from_rotvec(np.outer(np.radians(twc), a)).as_matrix())
            Lnew = np.einsum('ij,njk->nik', N, Rn)
            if parte in BRAZO and nom.startswith(('Left', 'Right')):
                w = libre['Left' if nom.startswith('Left') else 'Right']
                if w.min() < 0.999:
                    Lnew = s._mezclar(L[:, j], Lnew, w)
                exceso = exceso * w
            if exceso.max() < 0.5:
                continue
            s.limitado[nom] = round(float(exceso.max()), 1)
            L[:, j] = Lnew
        # ---- dedos (neutro = mano estirada de la tabla de manos de Max; doblez hacia la palma = la mano natural)
        tabla = np.load(os.path.join(AQUI, 'max_v7_mano_natural.npz'))
        # dirección de doblez de cada dedo: la de su articulación más doblada en la mano natural (la del medio); las
        # falanges de un dedo comparten esa dirección (en la mano estirada sus marcos casi coinciden)
        dobla = {}
        for nom, D, Nat in zip(tabla['nombres'], tabla['defecto'], tabla['natural']):
            nom = str(nom); rv = Ro.from_matrix(Nat @ D.T).as_rotvec(); dedo = nom[:-1]
            if np.linalg.norm(rv) > np.linalg.norm(dobla.get(dedo, np.zeros(3))):
                dobla[dedo] = rv
        # índice, medio, anular y meñique doblan sobre el mismo eje (la línea de los nudillos): el promedio de medio,
        # anular y meñique, que son los que la mano natural tiene bien doblados
        for lado in ('Left', 'Right'):
            ks = [lado + 'Hand' + d for d in ('Middle', 'Ring', 'Pinky') if lado + 'Hand' + d in dobla]
            if ks:
                ref = dobla[ks[-1]] / np.linalg.norm(dobla[ks[-1]])
                m_ = sum((dobla[k] / np.linalg.norm(dobla[k])) * np.sign(dobla[k] @ ref) for k in ks)
                for d in ('Index', 'Middle', 'Ring', 'Pinky'):
                    if lado + 'Hand' + d in dobla:
                        dobla[lado + 'Hand' + d] = m_ / np.linalg.norm(m_)
        for nom, D, Nat in zip(tabla['nombres'], tabla['defecto'], tabla['natural']):
            nom = str(nom)
            if nom not in clip.idx or nom not in s._ejes:
                continue
            j = clip.idx[nom]
            b = D @ s._ejes[nom]                              # eje del hueso (en el padre)
            u = dobla[nom[:-1]].copy(); u = u - b * (u @ b)
            if np.linalg.norm(u) < 1e-6:
                continue
            u /= np.linalg.norm(u); v = np.cross(b, u)
            R = np.einsum('nij,kj->nik', L[:, j], D)          # L @ D^T (en el padre)
            tw, sw = _giro_y_doblez(R, b)
            pulgar = 'Thumb' in nom
            gmax = s.PULGAR_GIRO if pulgar else s.DEDO_GIRO
            dlo, dhi = s.PULGAR_DOBLEZ if pulgar else s.DEDO_DOBLEZ
            lmax = s.PULGAR_LADO if pulgar else (s.DEDO_LADO[0] if nom[-1] in '12' else s.DEDO_LADO[1])
            twd = np.degrees(tw); fl = np.degrees(sw @ u); la = np.degrees(sw @ v)
            twc = np.clip(twd, -gmax, gmax); flc = np.clip(fl, dlo, dhi); lac = np.clip(la, -lmax, lmax)
            w = libre['Left' if nom.startswith('Left') else 'Right']
            ex = (np.maximum.reduce([np.abs(twd - twc), np.abs(fl - flc), np.abs(la - lac)]) * w).max()
            if ex < 0.5:
                continue
            s.limitado[nom] = round(float(ex), 1)
            swc = np.outer(np.radians(flc), u) + np.outer(np.radians(lac), v)
            Rn = np.einsum('nij,njk->nik', Ro.from_rotvec(swc).as_matrix(), Ro.from_rotvec(np.outer(np.radians(twc), b)).as_matrix())
            Lnew = np.einsum('nij,jk->nik', Rn, D)
            L[:, j] = s._mezclar(L[:, j], Lnew, w) if w.min() < 0.999 else Lnew
        nuevo = _rehacer(clip, L) if s.limitado else clip
        c[id(clip)] = (clip, nuevo)
        return nuevo

    @staticmethod
    def _mezclar(A, B, w):
        """Por cuadro: A donde w = 0, B donde w = 1 (suave entre medio)."""
        from scipy.spatial.transform import Rotation as Ro
        qa = Ro.from_matrix(A).as_quat(); qb = Ro.from_matrix(B).as_quat()
        qb = np.where(((qa * qb).sum(1) < 0)[:, None], -qb, qb)
        q = qa * (1 - w[:, None]) + qb * w[:, None]; q /= np.linalg.norm(q, axis=1, keepdims=True)
        return Ro.from_quat(q).as_matrix()

    def globales(s, clip, t, bucle=False, pies=True):
        return super().globales(s._limitar(clip), t, bucle, pies)


def _tope_suave(x, tope, rodilla=12.0):
    """Igual a x hasta (tope - rodilla); después se acerca SUAVE al tope sin pasarlo nunca."""
    a = np.abs(x); b = tope - rodilla
    return np.sign(x) * np.where(a < b, a, b + rodilla * np.tanh((a - b) / rodilla))


class _CuelloTotal:
    """Tope de la cabeza respecto al PECHO (sumando cuello y cabeza): girar a un lado como máximo ~70° (nunca mirando
    hacia atrás ni con la barbilla encima del hombro) y ladear la oreja hacia el hombro como máximo ~35°. Si la
    animación pide más hacia un lado, lo que sobra lo reparten el pecho (gira un poco, hasta 15°) y los ojos (miran más
    de lado, hasta 22°), como una persona de verdad."""
    CUELLO_TOTAL = True
    CUELLO_LADO_MAX = 70.0      # grados de giro de la cabeza a un lado respecto al pecho
    CUELLO_LADEO_MAX = 35.0     # grados de ladear la cabeza (oreja hacia el hombro)
    PECHO_GIRO_PARTE = 0.4      # parte de lo que sobra al girar que hace el pecho
    PECHO_GIRO_MAX = 15.0
    OJOS_LADO_MAX = 22.0
    # cabeza en el marco del pecho con Max mirando al frente (medido en el trote de Kimodo)
    _R0 = np.array([[0.0455, 0.02004, 0.99876], [0.8975, 0.43821, -0.04968], [-0.43867, 0.89865, 0.00195]])

    def _marco_pecho(s, clip):
        ix = clip.idx; P = clip.P
        up = P[:, ix['Neck1']] - P[:, ix['Chest']]; up /= np.linalg.norm(up, axis=1, keepdims=True)
        lat = P[:, ix['LeftShoulder']] - P[:, ix['RightShoulder']]
        lat -= up * (lat * up).sum(1, keepdims=True); lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        return np.stack([lat, up, np.cross(lat, up)], 2)

    def angulos_cuello(s, clip):
        """(giro a un lado, adelante/atrás, ladeo) de la cabeza respecto al pecho, en grados, por cuadro."""
        from scipy.spatial.transform import Rotation as Ro
        C = s._marco_pecho(clip)
        R = np.einsum('nji,njk->nik', C, clip.G[:, clip.idx['Head']])
        e = Ro.from_matrix(np.einsum('nij,kj->nik', R, s._R0)).as_euler('YXZ')
        e[:, 0] = np.unwrap(e[:, 0]); e[:, 2] = np.unwrap(e[:, 2])
        return np.degrees(e), C, R

    def _cuello_total(s, clip):
        if not s.CUELLO_TOTAL:
            return clip
        c = s.__dict__.setdefault('_clips_cuello', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.spatial.transform import Rotation as Ro
        ix = clip.idx; n = clip.n
        need = ('Chest', 'Neck1', 'Neck2', 'Head', 'Spine1', 'Spine2', 'LeftShoulder', 'RightShoulder')
        ojos = np.zeros(n)
        if not all(k in ix for k in need):
            c[id(clip)] = (clip, clip); return clip
        e, C, R = s.angulos_cuello(clip)
        yaw, pit, rol = e[:, 0], e[:, 1], e[:, 2]
        yc = _tope_suave(yaw, s.CUELLO_LADO_MAX); rc = _tope_suave(rol, s.CUELLO_LADEO_MAX)
        sobra = yaw - yc
        s.cuello_frenado = dict(lado=round(float(np.abs(sobra).max()), 1), ladeo=round(float(np.abs(rol - rc).max()), 1))
        if max(s.cuello_frenado.values()) < 0.5:
            c[id(clip)] = (clip, clip); s.__dict__.setdefault('_ojos_lado_por_clip', {})[id(clip)] = ojos; return clip
        # el pecho gira un poco hacia donde mira (no si una mano sostiene algo: movería el hombro)
        tt = np.arange(n) / clip.fps
        libre = np.ones(n)
        if hasattr(s, '_peso_agarre'):
            libre = np.array([1.0 - max(s._peso_agarre('Left', x), s._peso_agarre('Right', x)) for x in tt])
        pecho = np.sign(sobra) * np.minimum(np.abs(sobra) * s.PECHO_GIRO_PARTE, s.PECHO_GIRO_MAX) * libre
        ojos = np.sign(sobra) * np.minimum(np.abs(sobra - pecho), s.OJOS_LADO_MAX)
        Qc = Ro.from_euler('YXZ', np.radians(np.stack([yc, pit, rc], 1))).as_matrix()
        Rc = np.einsum('nij,jk->nik', Qc, s._R0)
        # donde hay que frenar, el cuello se rehace desde su postura neutra repartiendo el giro (35% abajo, 25% medio,
        # 40% la cabeza): así ninguna vértebra queda retorcida aunque la animación diera la vuelta entera
        tab = s._tabla_limites()
        N1, N2, N3 = (tab[k]['N'] for k in ('Neck1', 'Neck2', 'Head'))
        Gc = clip.G[:, ix['Chest']]
        T = np.einsum('nji,njk,nkl,ml->nim', Gc, C, Rc, N1 @ N2 @ N3)     # Gc^T (C Rc) (N1 N2 N3)^T
        rv = Ro.from_matrix(T).as_rotvec()
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        peso = _suave(np.maximum(np.abs(sobra), np.abs(rol - rc)) / 6.0)
        peso = np.clip(gaussian_filter1d(maximum_filter1d(peso, 11), 4.0, mode='nearest'), 0, 1)   # entra y sale suave
        L = clip.L.copy(); G = clip.G
        A = N1; B = N1 @ N2
        reh = {'Neck1': np.einsum('nij,jk->nik', Ro.from_rotvec(rv * 0.35).as_matrix(), N1),
               'Neck2': np.einsum('ji,njk,kl->nil', A, Ro.from_rotvec(rv * 0.25).as_matrix(), A @ N2),
               'Head': np.einsum('ji,njk,kl->nil', B, Ro.from_rotvec(rv * 0.4).as_matrix(), B @ N3)}
        for nom, Ln in reh.items():
            j = ix[nom]; L[:, j] = s._mezclar(L[:, j], Ln, peso)
        up = C[:, :, 1]
        for f in range(n):
            if abs(pecho[f]) > 1e-3:
                for nom, fr in (('Spine1', 0.3), ('Spine2', 0.35), ('Chest', 0.35)):
                    j = ix[nom]; pa = clip.padres[j]
                    L[f, j] = (G[f, pa].T @ _rot(up[f], np.radians(pecho[f] * fr)) @ G[f, pa]) @ L[f, j]
        nuevo = _rehacer(clip, L)
        c[id(clip)] = (clip, nuevo)
        s.__dict__.setdefault('_ojos_lado_por_clip', {})[id(clip)] = ojos
        return nuevo

    def globales(s, clip, t, bucle=False, pies=True):
        nuevo = s._cuello_total(clip)
        s._ojos_lado = s.__dict__.get('_ojos_lado_por_clip', {}).get(id(clip))
        s._ojos_lado_fps = clip.fps
        return super().globales(nuevo, t, bucle, pies)

    def pesos_cara(s, G, t, b=None):
        w = super().pesos_cara(G, t, b)
        o = getattr(s, '_ojos_lado', None)
        if s.CUELLO_TOTAL and o is not None and len(o):
            f = int(np.clip(round(t * getattr(s, '_ojos_lado_fps', 30.0)), 0, len(o) - 1))
            if abs(o[f]) > 0.2:
                from .cara import OJOS_LADO
                # + = hacia la izquierda de Max: ojo izquierdo hacia afuera, ojo derecho hacia adentro
                a, b_ = ('eyeLookOutLeft', 'eyeLookInRight') if o[f] > 0 else ('eyeLookInLeft', 'eyeLookOutRight')
                c_, d_ = ('eyeLookInLeft', 'eyeLookOutRight') if o[f] > 0 else ('eyeLookOutLeft', 'eyeLookInRight')
                k = abs(o[f]) / OJOS_LADO
                for x in (a, b_):
                    w[x] = float(np.clip(w.get(x, 0) + k, 0, 1))
                for x in (c_, d_):
                    w[x] = float(max(0.0, w.get(x, 0) - k))
        return w


class _MiradaRepartida:
    MIRADA_REPARTIDA = True
    CUELLO_ATRAS_MAX = 30.0     # grados que la cabeza se echa atrás respecto al pecho (lo que pase, se reparte)
    PECHO_PARTE = 0.5           # parte de lo que sobra que hace el pecho (echándose atrás)
    PECHO_MAX = 20.0
    OJOS_MAX = 20.0

    def _repartir(s, clip):
        if not s.MIRADA_REPARTIDA:
            return clip
        c = s.__dict__.setdefault('_clips_mirada', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.ndimage import gaussian_filter1d
        ix = clip.idx; P, G = clip.P, clip.G; n = clip.n
        need = ('Chest', 'Neck1', 'Neck2', 'Head', 'HeadEnd', 'Spine1', 'Spine2', 'LeftShoulder', 'RightShoulder')
        if not all(k in ix for k in need):
            c[id(clip)] = (clip, clip); return clip
        up = P[:, ix['Neck1']] - P[:, ix['Chest']]; up /= np.linalg.norm(up, axis=1, keepdims=True)
        lat = P[:, ix['LeftShoulder']] - P[:, ix['RightShoulder']]
        lat -= up * (lat * up).sum(1, keepdims=True); lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        fwd = np.cross(lat, up)
        hv = P[:, ix['HeadEnd']] - P[:, ix['Neck1']]; hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        atras = np.degrees(np.arctan2(-(hv * fwd).sum(1), (hv * up).sum(1)))
        # solo con el pecho derecho (mirar al cielo); agachado o acostado la cabeza levantada es a propósito
        incl = np.degrees(np.arccos(np.clip(up[:, 1], -1, 1)))
        derecho = 1 - _suave((incl - 20) / 15)
        e = gaussian_filter1d(np.maximum(atras - s.CUELLO_ATRAS_MAX, 0) * derecho, 3.0, mode='nearest')
        s.mirada_repartida = round(float(e.max()), 1)
        ojos = np.zeros(n)
        if e.max() < 0.5:
            c[id(clip)] = (clip, clip); s._ojos_arriba = ojos; return clip
        pecho = np.minimum(e * s.PECHO_PARTE, s.PECHO_MAX)
        ojos = np.minimum(e - pecho, s.OJOS_MAX)
        L = clip.L.copy()
        latw = P[:, ix['LeftShoulder']] - P[:, ix['RightShoulder']]; latw[:, 1] = 0
        latw /= np.linalg.norm(latw, axis=1, keepdims=True) + 1e-12
        for f in range(n):
            if e[f] < 1e-3:
                continue
            gp = -np.radians(pecho[f]); gc = np.radians(e[f])
            for nom, fr in (('Spine1', 0.3), ('Spine2', 0.35), ('Chest', 0.35)):     # el pecho atrás
                j = ix[nom]; pa = clip.padres[j]
                L[f, j] = (G[f, pa].T @ _rot(latw[f], gp * fr) @ G[f, pa]) @ L[f, j]
            for nom, fr in (('Neck1', 0.5), ('Neck2', 0.2), ('Head', 0.3)):          # el cuello menos atrás
                j = ix[nom]; pa = clip.padres[j]
                L[f, j] = (G[f, pa].T @ _rot(latw[f], gc * fr) @ G[f, pa]) @ L[f, j]
        nuevo = _rehacer(clip, L)
        c[id(clip)] = (clip, nuevo)
        s.__dict__.setdefault('_ojos_por_clip', {})[id(clip)] = ojos
        return nuevo

    def globales(s, clip, t, bucle=False, pies=True):
        nuevo = s._repartir(clip)
        s._ojos_arriba = s.__dict__.get('_ojos_por_clip', {}).get(id(clip))
        s._ojos_fps = clip.fps
        return super().globales(nuevo, t, bucle, pies)

    def pesos_cara(s, G, t, b=None):
        w = super().pesos_cara(G, t, b)
        o = getattr(s, '_ojos_arriba', None)
        if s.MIRADA_REPARTIDA and o is not None and len(o):
            f = int(np.clip(round(t * getattr(s, '_ojos_fps', 30.0)), 0, len(o) - 1))
            if o[f] > 0.2:
                from .cara import OJOS_ARRIBA
                for lado in ('Left', 'Right'):
                    w['eyeLookUp' + lado] = float(np.clip(w.get('eyeLookUp' + lado, 0) + o[f] / OJOS_ARRIBA, 0, 1))
                    w['eyeLookDown' + lado] = float(max(0.0, w.get('eyeLookDown' + lado, 0) - o[f] / OJOS_ARRIBA))
        return w


class _Seguimiento:
    SEGUIMIENTO = True
    # segundos de retraso de cada parte respecto a la de arriba (más = llega más tarde y más suave)
    RETRASO = {'Spine2': 0.02, 'Chest': 0.03, 'Neck1': 0.05, 'Neck2': 0.06, 'Head': 0.07,
               'Arm': 0.03, 'ForeArm': 0.045, 'Hand': 0.06}

    def _seguir(s, clip):
        if not s.SEGUIMIENTO:
            return clip
        c = s.__dict__.setdefault('_clips_seguidos', {})
        if id(clip) in c:
            return c[id(clip)][1]
        from scipy.spatial.transform import Rotation as Ro
        n = clip.n; fps = clip.fps; L = clip.L.copy()
        tt = np.arange(n) / fps
        libre = {lado: np.array([1.0 - s._peso_agarre(lado, t) for t in tt]) if hasattr(s, '_peso_agarre') else np.ones(n)
                 for lado in ('Left', 'Right')}
        for parte, tau in s.RETRASO.items():
            for nom in ([parte] if parte in clip.idx else ['Left' + parte, 'Right' + parte]):
                if nom not in clip.idx:
                    continue
                j = clip.idx[nom]
                q = Ro.from_matrix(L[:, j]).as_quat()
                for f in range(1, n):
                    if q[f] @ q[f - 1] < 0: q[f] = -q[f]
                al = 1 - np.exp(-1.0 / (fps * tau))
                qs = q.copy()
                for f in range(1, n):
                    x = qs[f - 1] + (q[f] - qs[f - 1]) * al; qs[f] = x / np.linalg.norm(x)
                if nom.startswith(('Left', 'Right')):
                    w = libre['Left' if nom.startswith('Left') else 'Right']
                elif nom in ('Spine1', 'Spine2', 'Chest'):        # el pecho mueve los hombros: quieto si una mano sostiene
                    w = np.minimum(libre['Left'], libre['Right'])
                else:
                    w = np.ones(n)
                qf = qs * w[:, None] + q * (1 - w[:, None]); qf /= np.linalg.norm(qf, axis=1, keepdims=True)
                L[:, j] = Ro.from_quat(qf).as_matrix()
        nuevo = _rehacer(clip, L)
        c[id(clip)] = (clip, nuevo)
        return nuevo

    def globales(s, clip, t, bucle=False, pies=True):
        return super().globales(s._seguir(clip), t, bucle, pies)


class MaxV16(_CuelloTotal, _Limites, _MiradaRepartida, _Seguimiento, MaxV15):
    """Max V16 sin cara viva (los ojos de la mirada repartida necesitan la cara viva)."""


class MaxCaraV16(_CuelloTotal, _Limites, _MiradaRepartida, _Seguimiento, MaxCaraV15):
    """Base de MaxCaraV17 (ZIP 43): Max V15 + límites del cuerpo + mirar arriba repartido + menos robótico."""
