"""ZIP 37 · Objetos de la escena junto a Max (para las pruebas y los renders de revisión).

escena6.py guarda <nombre>_objetos.npz (forma, medidas y dónde está cada objeto en cada cuadro, en el mundo de la
escena). Max es más alto que el esqueleto de la escena, así que su mundo no coincide exacto con el de la escena; aquí
cada objeto se coloca RESPECTO A LA MANO de Max:
  - mientras una mano lo sostiene, va pegado a esa mano igual que en la escena (con dos manos, al medio de las dos);
  - antes de agarrarlo, quieto donde la mano lo va a tomar; al soltarlo, quieto donde lo dejó;
  - lo que está debajo (la mesa de "sobre") va con su objeto; lo demás (muebles), con el cuerpo.
Uso:
    from soma.objetos_escena import ObjetosEscena
    obj = ObjetosEscena('<carpeta de salida de escena6.py>')
    obj.preparar(m, clip)               # una vez (m = MaxCaraV13 con su pista); recorre la animación
    piezas = obj.mallas(t)              # [(nombre, posiciones Nx3, normales Nx3, color RGB)] en el mundo de Max
"""
import numpy as np
from pathlib import Path

Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])        # escena (Y arriba) -> Max (Z arriba)
LADO = {'izq': 'Left', 'der': 'Right'}
COLORES = {'Cilindro': (0.20, 0.55, 0.30), 'Caja': (0.72, 0.52, 0.30), 'Esfera': (0.85, 0.35, 0.25),
           'Taza': (0.92, 0.92, 0.90), 'Aro': (0.80, 0.70, 0.20), 'Superficie': (0.55, 0.42, 0.30)}


def _marco_mano(P, L, g):
    """Marco de la mano (3x3) con las posiciones de sus huesos: hacia los dedos y a lo ancho de la palma."""
    h = g(P, L + 'Hand'); m1 = g(P, L + 'HandMiddle1'); i1 = g(P, L + 'HandIndex1'); p1 = g(P, L + 'HandPinky1')
    a = m1 - h; a /= np.linalg.norm(a) + 1e-12
    b = i1 - p1; b = b - a * (b @ a); b /= np.linalg.norm(b) + 1e-12
    return np.stack([a, b, np.cross(a, b)], 1), h


class ObjetosEscena:
    def __init__(s, carpeta):
        carpeta = Path(carpeta)
        f = next(carpeta.glob('*_objetos.npz'), None)
        s.ok = f is not None
        s.obj = []
        if not s.ok:
            return
        d = np.load(f, allow_pickle=True)
        s.fps = float(d['fps']); s.F = int(d['F'])
        for i, n in enumerate(d['nombres']):
            k = f'o{i}_'
            s.obj.append(dict(nombre=str(n), tipo=str(d[k + 'tipo']), medidas=d[k + 'medidas'], radio=float(d[k + 'radio']),
                              alto=float(d[k + 'alto']), tubo=float(d[k + 'tubo']), normal=d[k + 'normal'],
                              placa=d[k + 'placa'], sobre=str(d[k + 'sobre']), C=d[k + 'centro'].astype(float),
                              R=d[k + 'rot'].astype(float), sost={'izq': d[k + 'sost_izq'], 'der': d[k + 'sost_der']}))
        npz = next(carpeta.glob('*_L77.npz'))
        L = np.load(npz)
        import sys
        raiz = Path(__file__).resolve().parent.parent              # la carpeta del motor (donde está soma/)
        for extra in (carpeta.parent.parent, raiz / 'motionbricks_cpu', Path('/home/claude/motionbricks_cpu')):
            if (extra / 'agarres_ik.py').exists() and str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        import agarres_ik as A
        s._A = A
        s.L77, s.raiz = L['L77'], L['raiz']
        s.listo = False

    # ------------------------------------------------------------------------------------------- preparar
    def _escena_hueso(s, f):
        G, P = s._A.fk77(s.L77[min(f, len(s.L77) - 1)], s.raiz[min(f, len(s.raiz) - 1)])
        return P

    def preparar(s, m, clip, paso=1):
        """Recorre la animación con Max (m) y guarda, por cuadro, cómo pasar de la escena al mundo de Max:
        con cada mano (marco de la mano) y con el cuerpo (cadera)."""
        if not s.ok:
            return
        A = s._A
        gE = lambda P, n: Cz @ P[A.IDX[n]]
        gM = lambda P, n: m.joint_pos(n)
        n = s.F
        s.T = {'Left': [None] * n, 'Right': [None] * n, 'cuerpo': [None] * n}
        for f in range(0, n, paso):
            t = f / s.fps
            m.pose(clip, float(t))
            Pk = s._escena_hueso(f)
            for L in ('Left', 'Right'):
                Fk, hk = _marco_mano(Pk, L, gE)
                Fm, hm = _marco_mano(None, L, gM)
                R = Fm @ Fk.T
                s.T[L][f] = (R, hm - R @ hk)
            # cuerpo: giro de la cadera alrededor de la vertical + traslado
            hk = gE(Pk, 'Hips'); hm = m.joint_pos('Hips')
            s.T['cuerpo'][f] = (np.eye(3), hm - hk)
        for k in s.T:                                          # cuadros saltados: el anterior
            for f in range(n):
                if s.T[k][f] is None:
                    s.T[k][f] = s.T[k][f - 1]
        # pose de cada objeto en el mundo de Max, por cuadro
        for o in s.obj:
            o['Cm'] = np.zeros((n, 3)); o['Rm'] = np.zeros((n, 3, 3))
            sost = {LADO[k]: np.asarray(v, float)[:n] for k, v in o['sost'].items()}
            agarrado = np.maximum(sost['Left'], sost['Right']) > 0.97
            for f in range(n):
                C = Cz @ o['C'][f]; R = Cz @ o['R'][f] @ Cz.T
                if agarrado[f]:
                    usa = [L for L in ('Left', 'Right') if sost[L][f] > 0.97]
                    cs, rs = [], []
                    for L in usa:
                        Rt, tt = s.T[L][f]; cs.append(Rt @ C + tt); rs.append(Rt @ R)
                    o['Cm'][f] = np.mean(cs, 0)
                    U, _, Vt = np.linalg.svd(np.sum(rs, 0)); o['Rm'][f] = U @ Vt
                else:
                    o['Cm'][f] = np.nan
            # quieto antes de agarrar (donde lo toma) y después de soltar (donde lo deja)
            idx = np.where(agarrado)[0]
            if len(idx):
                for f in range(n):
                    if not agarrado[f]:
                        k = idx[np.argmin(np.abs(idx - f))]
                        # misma pose relativa a la escena: el objeto quieto en la escena -> quieto aquí
                        dk = Cz @ (o['C'][f] - o['C'][k])
                        o['Cm'][f] = o['Cm'][k] + dk; o['Rm'][f] = o['Rm'][k]
            else:
                for f in range(n):
                    Rt, tt = s.T['cuerpo'][f]
                    o['Cm'][f] = Cz @ o['C'][f] + tt; o['Rm'][f] = Cz @ o['R'][f] @ Cz.T
        # lo que está debajo de un objeto (la mesa) va con ese objeto
        por_nombre = {o['nombre']: o for o in s.obj}
        for o in s.obj:
            if o['sobre'] and o['sobre'] in por_nombre:
                base = por_nombre[o['sobre']]
                f0 = int(np.argmax(np.maximum(o['sost']['izq'], o['sost']['der']) > 0.97)) if \
                    np.maximum(o['sost']['izq'], o['sost']['der']).max() > 0.97 else 0
                d = Cz @ (base['C'][f0] - o['C'][f0])
                base['Cm'][:] = o['Cm'][f0] + d
                base['_fija'] = True
        # los muebles (sin "sobre" y nunca agarrados) quedan quietos donde estaban al empezar
        for o in s.obj:
            if not o.get('_fija') and np.maximum(o['sost']['izq'], o['sost']['der']).max() <= 0.97:
                o['Cm'][:] = o['Cm'][0]
        s.listo = True

    # ------------------------------------------------------------------------------------------- mallas
    def mallas(s, t):
        if not s.ok or not s.listo:
            return []
        f = int(np.clip(round(t * s.fps), 0, s.F - 1))
        out = []
        for o in s.obj:
            V, N = _malla(o)
            if V is None:
                continue
            C, R = o['Cm'][f], o['Rm'][f]
            if not np.all(np.isfinite(C)):
                continue
            out.append((o['nombre'], V @ R.T + C, N @ R.T, COLORES.get(o['tipo'], (0.7, 0.7, 0.7))))
        return out


# ----------------------------------------------------------------------------------------------- formas
def _cilindro(r, h, n=28, y0=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V, N = [], []
    for i in range(n):
        a0, a1 = a[i], a[(i + 1) % n]
        p = lambda ang, y: np.array([r * np.cos(ang), y, r * np.sin(ang)])
        q = lambda ang: np.array([np.cos(ang), 0, np.sin(ang)])
        V += [p(a0, y0), p(a1, y0), p(a1, y0 + h), p(a0, y0), p(a1, y0 + h), p(a0, y0 + h)]
        N += [q(a0), q(a1), q(a1), q(a0), q(a1), q(a0)]
        V += [np.array([0, y0 + h, 0]), p(a1, y0 + h), p(a0, y0 + h)]; N += [np.array([0, 1., 0])] * 3
        V += [np.array([0, y0, 0]), p(a0, y0), p(a1, y0)]; N += [np.array([0, -1., 0])] * 3
    return np.array(V), np.array(N)


def _caja(mx, my, mz):
    hx, hy, hz = mx / 2, my / 2, mz / 2
    V, N = [], []
    for ax in range(3):
        for sg in (-1, 1):
            n = np.zeros(3); n[ax] = sg
            u = np.zeros(3); u[(ax + 1) % 3] = 1; v = np.cross(n, u)
            h = np.array([hx, hy, hz])
            c = n * h[ax]
            du = u * h[(ax + 1) % 3]; dv = v * np.abs(v) @ h * np.sign(v.sum()) if False else v * h[(ax + 2) % 3]
            p = [c - du - dv, c + du - dv, c + du + dv, c - du + dv]
            V += [p[0], p[1], p[2], p[0], p[2], p[3]]; N += [n] * 6
    return np.array(V), np.array(N)


def _esfera(r, n=16):
    V, N = [], []
    for i in range(n):
        for j in range(2 * n):
            def pt(a, b):
                return np.array([np.sin(a) * np.cos(b), np.cos(a), np.sin(a) * np.sin(b)])
            a0, a1 = np.pi * i / n, np.pi * (i + 1) / n; b0, b1 = np.pi * j / n, np.pi * (j + 1) / n
            q = [pt(a0, b0), pt(a1, b0), pt(a1, b1), pt(a0, b1)]
            for tri in ((0, 1, 2), (0, 2, 3)):
                for k in tri:
                    V.append(q[k] * r); N.append(q[k])
    return np.array(V), np.array(N)


def _malla(o):
    t = o['tipo']
    if t == 'Cilindro':
        V, N = _cilindro(o['radio'], o['alto'])          # el centro de la escena es la base del cilindro
    elif t == 'Caja':
        V, N = _caja(*o['medidas'])
    elif t == 'Esfera':
        V, N = _esfera(o['radio'])
    elif t == 'Taza':
        V, N = _cilindro(0.04, 0.095)
    elif t == 'Superficie':
        r = max(o['radio'], 0.25)
        V1, N1 = _cilindro(r, 0.03, 36, y0=-0.03)                                # tablero
        V2, N2 = _cilindro(0.03, 0.9, 12, y0=-0.93)                              # pata (hasta el piso, aprox.)
        V, N = np.r_[V1, V2], np.r_[N1, N2]
    else:
        return None, None
    # escena (Y arriba) -> Max (Z arriba)
    return V @ Cz.T, N @ Cz.T
