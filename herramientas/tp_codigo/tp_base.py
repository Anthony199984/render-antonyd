"""Base común del video "Si pudieras teletransportarte": rutas, tiempos de la voz, caras por tiempo de escena,
dos Max a la vez, cámaras y efectos de imagen (rebobinado, desenfoque de barrido, fundidos).

Cada toma (tNN.py) define una clase con INI, FIN (segundos del video final), preparar(E), avanzar(t) (estado, opcional)
y cuadro(E, t) -> imagen uint8 HxWx3."""
import sys, os, math, numpy as np
for p in ('/home/claude/ciudad_lovely', '/home/claude/teletransporte', '/home/claude'):
    if p not in sys.path: sys.path.insert(0, p)
import lovely as L
import hq, soma.sin_fugas  # noqa: F401  (render sin fuga de memoria)
from soma import Clip
from soma.cara import PistaCara
B = L.B
RAIZ = '/home/claude/teletransporte'
GESTOS = '/home/claude/biblioteca_gestos'
ANIMS = '/home/claude/biblioteca_animaciones'
TP = RAIZ + '/anims'
OBJ = '/home/claude/recursos_repo/recursos/out/objaverse'
SON = '/home/claude/recursos_repo/recursos/out/sonidos'
PAUSA_EN, PAUSA = 60.3, 1.0              # en la voz final se metió 1 s de silencio después de "se va la luz"
FIN_VIDEO = 89.2


def vt(t):
    """Segundo de la voz original (palabras.json) -> segundo del video final."""
    return t + (PAUSA if t >= PAUSA_EN else 0.0)


def ruta_anim(n):
    for d in (TP, GESTOS, ANIMS):
        p = f'{d}/{n}.bvh'
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


def suave(u):
    u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)


def mezcla(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


def claves(u, ks):
    return B.claves(min(max(u, 0.0), 1.0), ks)


class Pista(PistaCara):
    """Pista de caras en el tiempo de la ESCENA (no del clip): antes de cada pose se pone .ahora."""
    ahora = 0.0

    def pesos(s, t):
        return PistaCara.pesos(s, s.ahora)


def caras(tramos, semilla=7):
    return Pista(tramos, semilla=semilla)


def sin_puntos(img):
    """Quita los puntitos negros sueltos (píxeles NaN del sombreado de la piel con sol bajo): cada píxel casi negro
    rodeado de claros toma la mediana de sus vecinos. No toca sombras ni negros de verdad (que no están aislados)."""
    from scipy.ndimage import uniform_filter, median_filter
    s_ = img.astype(np.int32).sum(2)
    osc = (s_ < 45).astype(np.float32)
    m = (s_ < 45) & (uniform_filter(s_.astype(np.float32), 7) > 140) & (uniform_filter(osc, 7) < .30)   # negro, chiquito y rodeado de claro
    if not m.any(): return img
    med = median_filter(img, size=(7, 7, 1))
    out = img.copy(); out[m] = med[m]
    return out


def _vida(E, cam):
    """Cámara en mano muy suave: la posición y el punto de mira se mecen unos milímetros (nunca queda clavada).
    Continua en el tiempo, así que no mete saltos dentro de una toma."""
    t = getattr(E, 't_escena', None)
    if t is None or getattr(E, 'sin_vida', False): return cam
    ojo, mira, fov = np.asarray(cam[0], float), np.asarray(cam[1], float), cam[2]
    d = float(np.linalg.norm(mira - ojo)); amp = .004 + .005 * min(d, 3.0)
    n = lambda k: (math.sin(t * 1.43 + k) * .6 + math.sin(t * 2.71 + 2 * k) * .3 + math.sin(t * 4.1 + 3 * k) * .1)
    ojo = ojo + amp * np.array([n(0), n(1.7), n(3.1) * .6]); mira = mira + amp * .5 * np.array([n(4.2), n(5.3), n(6.6) * .6])
    return (tuple(ojo), tuple(mira), fov) + tuple(cam[3:])


def escena(W, H):
    E = L.Escena(W, H); E.max.pista = caras([(-5, 999, 'neutral')])
    E.cam_mod = lambda cam: _vida(E, cam)
    r0 = E.render
    E.render = lambda *a, **k: sin_puntos(r0(*a, **k))
    return E


def clip(E, n):
    if n not in E.clips: E.clips[n] = Clip(ruta_anim(n))
    return E.clips[n]


def max_objs(E, n, tc, pos, yaw, pista=None, t_escena=0.0, suelo=None, otro=False, ancla_t=0.0, ik=None):
    """Max con el clip n en el tiempo tc, la cara de 'pista' en el tiempo de escena; 'otro' = segundo Max (mallas aparte)."""
    clip(E, n)
    import ik_tp
    ik_tp.instalar(E.max); E.max.ik = ik
    try:
        return _max_objs(E, n, tc, pos, yaw, pista, t_escena, suelo, otro, ancla_t)
    finally:
        E.max.ik = None


def _max_objs(E, n, tc, pos, yaw, pista, t_escena, suelo, otro, ancla_t):
    if pista is not None: E.max.pista = pista
    E.max.pista.ahora = t_escena
    zs = getattr(E, 'zsuelo', 0.0)
    if suelo is not None: E.zsuelo = suelo
    if otro:
        import real_scene as RS
        orig = RS.set_mesh
        RS.set_mesh = lambda r, nm, d: orig(r, nm.replace('max_m', 'max2_m').replace('sombra', 'sombra2'), d)
        try:
            E.max._ult_pose = None
            o, M, J = E.max_objs(n, tc, tuple(pos), yaw, ancla_t=ancla_t)
        finally:
            RS.set_mesh = orig
        for d in o:
            d['mesh'] = d['mesh'].replace('max_m', 'max2_m').replace('sombra', 'sombra2')
        E.max._ult_pose = None
    else:
        o, M, J = E.max_objs(n, tc, tuple(pos), yaw, ancla_t=ancla_t)
    E.zsuelo = zs
    return o, M, J


# ------------------------------------------------------------------ efectos de imagen
def desenfoque_h(img, px):
    """Desenfoque horizontal (barrido) de px píxeles."""
    if px < 1: return img
    a = img.astype(np.float32); k = int(px); out = np.zeros_like(a); n = 0
    for d in range(-k, k + 1, max(1, k // 6)):
        out += np.roll(a, d, axis=1); n += 1
    return np.clip(out / n, 0, 255).astype(np.uint8)


def desenfoque_v(img, px):
    if px < 1: return img
    a = img.astype(np.float32); k = int(px); out = np.zeros_like(a); n = 0
    for d in range(-k, k + 1, max(1, k // 6)):
        out += np.roll(a, d, axis=0); n += 1
    return np.clip(out / n, 0, 255).astype(np.uint8)


def rebobinado(img, fuerza, t):
    """Look de cinta rebobinando: desaturado frío, líneas que corren hacia abajo y un poco de barrido."""
    a = img.astype(np.float32); H, W = a.shape[:2]
    g = a.mean(2, keepdims=True); a = a * (1 - .45 * fuerza) + g * .45 * fuerza
    a *= np.array([.92, .97, 1.08]) ** fuerza
    y = (np.arange(H) / H * 9 + t * 7.0) % 1.0
    banda = (np.abs(y - .5) < .035).astype(np.float32)[:, None, None] * .35 * fuerza
    a = a * (1 - banda) + 255 * banda * .6
    off = (np.sin(np.arange(H) * .05 + t * 40) * 6 * fuerza).astype(int)
    for k in range(0, H, 8): a[k:k + 8] = np.roll(a[k:k + 8], off[k], axis=1)
    return desenfoque_v(np.clip(a, 0, 255).astype(np.uint8), 3 * fuerza)


def fundido(a, b, u):
    return (a.astype(np.float32) * (1 - u) + b.astype(np.float32) * u).astype(np.uint8)


def oscurecer(img, k):
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)


_COL = {}


def colocar(E, n, tc, objetivo, yaw, ancla_t=0.0):
    """pos para max_objs tal que la cadera del clip n en el tiempo tc quede sobre 'objetivo' (x, y)."""
    k = (n, round(tc, 3), round(yaw, 2), ancla_t)
    if k not in _COL:
        c = clip(E, n); m = E.max; ix = m.idx
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
        P0 = m.globales(c, ancla_t, pies=False)[1][ix['Hips']]; P1 = m.globales(c, tc, pies=False)[1][ix['Hips']]
        d = ((Cz @ P1) - (Cz @ P0)) * m.base.scale
        th = math.radians(yaw); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        _COL[k] = R @ d[:2]
    v = _COL[k]
    return np.array([objetivo[0] - v[0], objetivo[1] - v[1], objetivo[2] if len(objetivo) > 2 else 0.0])


def mano_mat(E, M, lado='Left'):
    """Marco de la mano de Max tras el último max_objs (mundo): x = hacia los dedos, y = del meñique al índice, z = normal.
    Sale de las articulaciones ya deformadas (incluye las correcciones del motor), para pegar objetos que la mano sostiene."""
    j = lambda n: (M @ np.r_[E.max.joint_pos(lado + n), 1])[:3]
    h = j('Hand'); x = j('HandMiddle2') - h; x /= np.linalg.norm(x)                 # igual que ik_tp.marco0 (nudillos)
    y = j('HandIndex2') - j('HandPinky2'); y = y - x * (y @ x); y /= np.linalg.norm(y)
    W = np.eye(4); W[:3, 0] = x; W[:3, 1] = y; W[:3, 2] = np.cross(x, y); W[:3, 3] = h
    return W


def boca(E, M):
    """Centro de la boca de Max (mundo) tras el último max_objs."""
    a = (M @ np.r_[E.max.joint_pos('Head'), 1])[:3]; jw = (M @ np.r_[E.max.joint_pos('Jaw'), 1])[:3]
    return jw + (jw - a) * .0


def objeto_en(r, nombre, W, local_pos, local_R, largo=None, alto=None):
    """Modelo pegado a un marco W (4x4 mundo) con posición/rotación locales."""
    import recursos_vel as RV
    R = W[:3, :3] @ local_R; p = W[:3, :3] @ np.asarray(local_pos, float) + W[:3, 3]
    return RV.modelo(r, nombre, tuple(p), 0.0, largo=largo, alto=alto, R=R)


def verts_modelo(nombre, objs):
    """Vértices en el mundo de un modelo ya colocado con RV.modelo (objs = lo que devolvió)."""
    import recursos_vel as RV
    M = np.asarray(objs[0]['model'], float).reshape(4, 4)
    return np.concatenate([np.asarray(q['P'], float) @ M[:3, :3].T + M[:3, 3] for q in RV._partes(nombre)['partes']])


def tope(V, xy, radio=.08, zmax=1.4, zmin=.3):
    """Altura de la superficie más alta (entre zmin y zmax) cerca de xy en una nube de vértices; None si no hay."""
    d = np.hypot(V[:, 0] - xy[0], V[:, 1] - xy[1]); s_ = (d < radio) & (V[:, 2] < zmax) & (V[:, 2] > zmin)
    return float(V[s_, 2].max()) if s_.any() else None


# ---------------------------------------------------------------------------------------------- resaltado (estilo Zack D. Films)
CIAN = np.array((90, 235, 255), np.float32)
DORADO = np.array((255, 214, 140), np.float32)


def rampa(t, t0, entra=.1, dura=.8, sale=.18):
    """Intensidad del resaltado: entra en ~3 cuadros, se queda y sale suave."""
    if t < t0 or t > t0 + dura + sale: return 0.0
    if t < t0 + entra: return (t - t0) / entra
    if t < t0 + dura: return 1.0
    x = (t - t0 - dura) / sale; return 1 - x * x * (3 - 2 * x)


def render_resaltado(E, objs, cam, L, es, k, col=CIAN, crema=.28):
    """Dibuja la escena y, si k > 0, resalta los objetos para los que es(d) es True: se aclaran, borde brillante y halo.
    La máscara sale de comparar la profundidad con y sin esos objetos (respeta lo que los tapa)."""
    img = E.render(objs, cam, L)
    if k <= 0: return img
    import cv2
    pc = np.asarray(E.prof, np.float32).copy()
    E.r.draw([d for d in objs if not es(d)], getattr(E, 'cam_usada', cam), L)
    ps = np.asarray(E.r.lin_depth(L['near'], L['far']), np.float32)
    m = (pc < ps - 1e-3).astype(np.float32)
    if m.shape[:2] != img.shape[:2]: m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return resaltar(img, m, k, col, crema)


def resaltar(img, m, k, col=CIAN, crema=.28):
    import cv2
    if k <= 0 or m.max() <= 0: return img
    m = np.clip(m, 0, 1).astype(np.float32); out = img.astype(np.float32)
    out = out * (1 - m[..., None] * crema * k) + m[..., None] * np.array((255, 246, 228)) * crema * k
    H = img.shape[0]; s = H / 960.
    ker = np.ones((3, 3), np.uint8); it = max(1, int(round(2 * s)))
    edge = cv2.dilate(m, ker, iterations=it) - cv2.erode(m, ker, iterations=max(1, it // 2))
    glow = cv2.GaussianBlur(cv2.dilate(m, ker, iterations=int(3 * s) + 1), (0, 0), 9 * s) + cv2.GaussianBlur(m, (0, 0), 26 * s) * .8
    glow = np.clip(glow - m * .7, 0, 1)
    glow = glow * (1 - m * .9)                                          # el halo va por fuera: el objeto no se quema a blanco
    out += (glow[..., None] * .80 + np.clip(edge, 0, 1)[..., None] * .85) * np.asarray(col, np.float32) * k
    return np.clip(out, 0, 255).astype(np.uint8)
