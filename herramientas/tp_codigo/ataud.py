"""Caja de muertos (ataúd) por código, de pie, en dos mitades que se cierran alrededor de Max.
Perfil hexagonal clásico (angosto en los pies, ancho en los hombros), madera de nogal laqueada, forro de satén crema,
tapa con panel en relieve y asas de bronce. Marco local: x a lo ancho, y en la profundidad (la tapa mira a -y, igual que
Max con giro 0), z hacia arriba (z = 0 en el piso).
    import ataud as AT
    AT.preparar(E.r)
    objs += AT.objetos(u, centro, giro)      # u: 0 = mitades lejos, 1 = cerrada
    AT.dentro(V_mundo, centro, giro)         # cuánto se sale cada vértice de Max del interior (m; > 0 = atraviesa)"""
import math, numpy as np
from PIL import Image, ImageFilter
import lovely_vida as V

Z = (0.0, 1.42, 2.08)
W = (.33, .43, .30)           # medio ancho por fuera a esa altura (cabe Max con los brazos y los tenis)
D = .60                          # profundidad total
T = .028                         # grosor de la madera
ESC = 1.0
BAJO = -.036                     # el fondo queda hundido en el piso: Max pisa el piso de la calle, no la tabla
MADERA = dict(roughness=.24, metal=.05)
FORRO = dict(roughness=.40, metal=.08)
BRONCE = dict(roughness=.28, metal=.85)


def _hex(sc=1.0, z0=0.0):
    z = [z0 + (Z[0] - z0) * 1, Z[1], Z[2]]
    return [np.array(p, float) for p in ((-W[0] * sc, Z[0]), (W[0] * sc, Z[0]), (W[1] * sc, Z[1]), (W[2] * sc, Z[2]), (-W[2] * sc, Z[2]), (-W[1] * sc, Z[1]))]


def _hex_interior(m=T):
    """Hexágono interior (offset de cada lado hacia adentro m metros)."""
    H = _hex(); n = len(H); L = []
    for i in range(n):
        a, b = H[i], H[(i + 1) % n]; d = (b - a) / np.linalg.norm(b - a); nin = np.array([-d[1], d[0]])     # antihorario: adentro a la izquierda
        L.append((a + nin * m, d))
    P = []
    for i in range(n):
        (p1, d1), (p2, d2) = L[i - 1], L[i]
        A = np.array([d1, -d2]).T; s = np.linalg.solve(A, p2 - p1); P.append(p1 + d1 * s[0])
    return P


def _tex_madera(r):
    if 'ataud_madera3' in r.textures: return
    rg = np.random.default_rng(5); N = 512
    def ruido(esc, sem):
        g = np.random.default_rng(sem).random((N // esc + 2, N // esc + 2))
        return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((N, N), Image.BICUBIC), float) / 255
    n = ruido(128, 1) * .7 + ruido(32, 2) * .3
    u = np.linspace(0, 1, N)[None, :].repeat(N, 0); v = np.linspace(0, 1, N)[:, None].repeat(N, 1)
    veta = .5 + .5 * np.sin((u * 11 + n * 3.2 + np.sin(v * 5) * .25) * math.pi * 2)     # veta larga y suave (antes parecía aglomerado)
    veta = veta ** 4 * .55
    fibra = (np.asarray(Image.fromarray((np.random.default_rng(6).random((N, 8)) * 255).astype(np.uint8)).resize((N, N), Image.BICUBIC), float) / 255 - .5) * .10
    a = np.array([206, 176, 136.0]); b = np.array([246, 230, 202.0])          # arce claro laqueado: destaca de noche
    im = a + (b - a) * np.clip(veta + fibra, 0, 1)[..., None]
    img = Image.fromarray(im.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6)).convert('RGBA')
    r.add_texture('ataud_madera3', img); r.set_texture_filter('ataud_madera3', 'linear')


def _p3(xz, y): return np.array([xz[0], y, xz[1]])


def _mitad(lado):
    """lado = +1: mitad de atrás (fondo + paredes de y 0 a +D/2); -1: la de adelante (tapa + paredes de -D/2 a 0)."""
    Mm = V.Malla(); Mf = V.Malla(); Mb = V.Malla(); BLANCO = (1, 1, 1); CREMA = (.30, .035, .05)           # forro de satén rojo vino (contrasta con la madera clara)
    H = _hex(); Hi = _hex_interior(); n = len(H)
    y_ext = lado * D / 2; y_int = lado * (D / 2 - T); y_jun = 0.0
    def uvq(a, b, c, d, eje_u, eje_v):          # uv en metros (la veta corre a lo largo de eje_v)
        return tuple((float(p @ eje_u) * 1.1, float(p @ eje_v) * .55) for p in (a, b, c, d))
    def q(Mx, a, b, c, d, col, out, plano=False):
        """Quad con la normal hacia 'out' (da vuelta el orden si hace falta)."""
        nn = np.cross(b - a, c - a)
        if nn @ out < 0: a, b, c, d = b, a, d, c
        ex = (b - a) / max(np.linalg.norm(b - a), 1e-9); ey = np.cross(np.cross(b - a, c - a), ex); ey /= max(np.linalg.norm(ey), 1e-9)
        if plano: ex, ey = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])     # tapa y fondo: la veta va a lo largo de la caja
        Mx.quad(a, b, c, d, col, uvq(a, b, c, d, ex, ey) if Mx is Mm else ((0, 0), (1, 0), (1, 1), (0, 1)))
    # paredes (6 lados): por fuera madera, por dentro forro, cantos de madera
    for i in range(n):
        a, b = H[i], H[(i + 1) % n]; ai, bi = Hi[i], Hi[(i + 1) % n]
        mid = (a + b) / 2; out2 = mid - (ai + bi) / 2; out = np.array([out2[0], 0, out2[1]])
        q(Mm, _p3(a, y_jun), _p3(b, y_jun), _p3(b, y_ext), _p3(a, y_ext), BLANCO, out)                 # cara de afuera
        q(Mf, _p3(ai, y_jun), _p3(bi, y_jun), _p3(bi, y_int), _p3(ai, y_int), CREMA, -out)            # forro
        q(Mm, _p3(a, y_jun), _p3(b, y_jun), _p3(bi, y_jun), _p3(ai, y_jun), BLANCO, np.array([0, -lado, 0.0]))   # canto de la junta
    # fondo (o tapa): hexágono por fuera y por dentro
    c = sum(H) / n; ci = sum(Hi) / n
    for i in range(n):
        a, b = H[i], H[(i + 1) % n]; ai, bi = Hi[i], Hi[(i + 1) % n]
        q(Mm, _p3(c, y_ext), _p3(a, y_ext), _p3(b, y_ext), _p3(b, y_ext), BLANCO, np.array([0, lado, 0.0]), plano=True)
        q(Mf, _p3(ci, y_int), _p3(ai, y_int), _p3(bi, y_int), _p3(bi, y_int), CREMA, np.array([0, -lado, 0.0]), plano=True)
    if lado < 0:                                  # tapa con panel en relieve (biselado) y moldura
        y1 = y_ext - .016; Hp = [c + (p - c) * .80 for p in H]; Hq = [c + (p - c) * .74 for p in H]
        for i in range(n):
            a, b = Hp[i], Hp[(i + 1) % n]; a2, b2 = Hq[i], Hq[(i + 1) % n]
            q(Mm, _p3(a, y_ext - .001), _p3(b, y_ext - .001), _p3(b2, y1), _p3(a2, y1), BLANCO, np.array([0, -1.0, 0]) + np.array([(a - c)[0], 0, (a - c)[1]]) * .2)
            q(Mm, _p3(c, y1), _p3(a2, y1), _p3(b2, y1), _p3(b2, y1), BLANCO, np.array([0, -1.0, 0]), plano=True)
    else:                                         # asas de bronce en los dos costados largos (van con la mitad de atrás)
        for (a, b) in ((H[1], H[2]), (H[5], H[0])):
            d = b - a; L = np.linalg.norm(d); d /= L; nout = np.array([d[1], -d[0]])
            if nout @ ((a + b) / 2 - c) < 0: nout = -nout
            for f in (.22, .52, .80):
                p = a + d * L * f; y = D / 4
                p0 = _p3(p + nout * .045, y) - np.array([0, .07, 0]); p1 = _p3(p + nout * .045, y) + np.array([0, .07, 0])
                Mb.cil(p0, p1, .011, (1, 1, 1), seg=8, tapas=True)
                for e in (p0, p1): Mb.cil(e, e - np.array([nout[0], 0, nout[1]]) * .045, .008, (1, 1, 1), seg=6)
    return Mm, Mf, Mb


def preparar(r):
    _tex_madera(r)
    if 'ataud_atras_m' in r.meshes: return
    for lado, nom in ((1, 'atras'), (-1, 'frente')):
        Mm, Mf, Mb = _mitad(lado)
        Mm.subir(r, f'ataud_{nom}_m'); Mf.subir(r, f'ataud_{nom}_f')
        if Mb.P: Mb.subir(r, f'ataud_{nom}_b')


def _R(yaw):
    a = math.radians(yaw); return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def objetos(u, centro, giro, sacudon=0.0):
    """u: 0 = las dos mitades lejos (fuera de cuadro, en esquinas opuestas), 1 = cerrada alrededor de 'centro' (piso, debajo de la cadera)."""
    e = min(1.0, max(0.0, u)) ** 1.6                         # acelera y se cierra de golpe
    R0 = _R(giro); c = np.asarray(centro, float) + (0, 0, BAJO); o = []
    for nom, lado in (('atras', 1), ('frente', -1)):
        lejos = 1 - e
        off_l = np.array([-1.45 * lado, 1.35 * lado, .30]) * lejos          # de esquinas opuestas
        rot = _R(-75 * lado * lejos) @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        piv = np.array([0, 0, 1.0])                                         # gira sobre su propio centro
        M = np.eye(4, dtype=np.float32)
        Rl = rot; pl = piv - Rl @ piv + off_l
        if sacudon: pl = pl + np.array([0, 0, .012 * sacudon])
        M[:3, :3] = R0 @ Rl * ESC; M[:3, 3] = c + R0 @ pl * ESC
        o.append(dict(mesh=f'ataud_{nom}_m', model=M, texture='ataud_madera3', color=(1, 1, 1), **MADERA))
        o.append(dict(mesh=f'ataud_{nom}_f', model=M.copy(), texture='white', color=(1, 1, 1), **FORRO))
        if nom == 'atras': o.append(dict(mesh='ataud_atras_b', model=M.copy(), texture='white', color=(.86, .66, .32), **BRONCE))
    return o


def dentro(Pw, centro, giro, margen=.006):
    """Para cada punto del mundo: cuánto se sale del interior de la caja cerrada (m). <= 0 = adentro."""
    Pl = (np.asarray(Pw, float) - np.asarray(centro, float) - (0, 0, BAJO)) @ _R(giro) / ESC
    Hi = _hex_interior(T + margen); n = len(Hi); fuera = np.full(len(Pl), -9.0)
    xz = Pl[:, [0, 2]]
    for i in range(n):
        a, b = Hi[i], Hi[(i + 1) % n]; d = (b - a) / np.linalg.norm(b - a); nout = np.array([d[1], -d[0]])
        fuera = np.maximum(fuera, (xz - a) @ nout)
    fuera = np.maximum(fuera, np.abs(Pl[:, 1]) - (D / 2 - T - margen))
    return fuera
