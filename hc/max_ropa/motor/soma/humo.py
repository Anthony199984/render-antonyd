"""ZIP 39 · HUMO REAL POR CÓDIGO (bocanadas que se disipan), dibujado encima del render de hq con su profundidad.

Partículas suaves (sin bolitas): cada una es una nube de ruido fractal con bordes difusos que crece, se mueve con el
aire (frena rápido: el que trota la deja atrás), sube un poco, gira con remolinos y se apaga. Muchas juntas hacen una
bocanada densa. Respeta lo que está delante (el cuerpo tapa el humo de atrás) y se funde suave al tocar las cosas.

    from soma.humo import Humo
    h = Humo()
    h.emitir(t, puntos, normales, velocidad, n, **ajustes)     # en el segundo t, n partículas desde esos puntos
    h.avanzar(t)                                               # mueve todo hasta el segundo t
    hdr = h.dibujar(r, hdr, (eye, target, fov))                # después de r.draw(...), antes de hq.develop(...)
"""
import numpy as np
import cv2


def _ruido_fractal(n, semilla):
    rng = np.random.default_rng(semilla)
    img = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for oct_ in range(5):
        k = 4 * 2 ** oct_
        base = rng.random((k, k)).astype(np.float32)
        img += amp * cv2.resize(base, (n, n), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.55
    img /= tot
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


class Humo:
    COLOR = (0.78, 0.79, 0.82)        # gris claro (polvo/humo)
    LUZ_ARRIBA = 0.5                 # la parte de arriba de cada nube, más clara (luz del cielo)
    TAU = 0.28                        # s: lo rápido que el aire frena la partícula
    SUBE = 0.18                       # m/s hacia arriba (aire caliente)
    REMOLINO = 0.55                   # m/s de los remolinos
    TEX = 96

    def __init__(s, semilla=7):
        s.rng = np.random.default_rng(semilla)
        s.P = np.zeros((0, 3)); s.V = np.zeros((0, 3))
        s.nace = np.zeros(0); s.vida = np.zeros(0); s.r0 = np.zeros(0); s.r1 = np.zeros(0); s.a0 = np.zeros(0)
        s.tex_i = np.zeros(0, int); s.giro = np.zeros(0); s.tono = np.zeros(0)
        s.t = 0.0
        # texturas: nube de ruido con borde suave
        yy, xx = np.mgrid[-1:1:complex(s.TEX), -1:1:complex(s.TEX)]
        rr = np.sqrt(xx ** 2 + yy ** 2)
        borde = np.clip(1 - rr, 0, 1) ** 1.6
        s.tex = []
        for k in range(10):
            n = _ruido_fractal(s.TEX, semilla * 31 + k)
            a = np.clip(borde * (0.35 + 0.95 * n) - 0.08, 0, 1)
            s.tex.append(a.astype(np.float32))
        s.grad = np.linspace(1 + s.LUZ_ARRIBA, 1 - s.LUZ_ARRIBA * 0.6, s.TEX, dtype=np.float32)[:, None]
        # remolinos: unas ondas al azar (campo suave que cambia con el tiempo)
        s.modos = [(s.rng.normal(size=3) * 3.5, s.rng.normal(size=3), s.rng.uniform(0, 6.28), s.rng.uniform(0.8, 2.0))
                   for _ in range(6)]

    def emitir(s, t, puntos, normales, velocidad, n, r0=(0.04, 0.07), r1=(0.16, 0.30), vida=(0.75, 1.25),
               alfa=(0.22, 0.38), sale=0.6, azar=0.25):
        if n <= 0 or len(puntos) == 0:
            return
        k = s.rng.integers(0, len(puntos), n)
        p = puntos[k] + s.rng.normal(scale=0.015, size=(n, 3))
        v = np.asarray(velocidad, float) + normales[k] * sale * s.rng.uniform(0.5, 1.2, (n, 1)) + \
            s.rng.normal(scale=azar, size=(n, 3))
        U = lambda ab: s.rng.uniform(ab[0], ab[1], n)
        s.P = np.r_[s.P, p]; s.V = np.r_[s.V, v]; s.nace = np.r_[s.nace, np.full(n, t) + s.rng.uniform(0, 1 / 30, n)]
        s.vida = np.r_[s.vida, U(vida)]; s.r0 = np.r_[s.r0, U(r0)]; s.r1 = np.r_[s.r1, U(r1)]; s.a0 = np.r_[s.a0, U(alfa)]
        s.tex_i = np.r_[s.tex_i, s.rng.integers(0, len(s.tex), n)]; s.giro = np.r_[s.giro, s.rng.uniform(0, 360, n)]
        s.tono = np.r_[s.tono, s.rng.uniform(0.78, 1.08, n)]          # unas nubes más grises que otras (volumen)

    def _remolino(s, P, t):
        u = np.zeros_like(P)
        for k, d, ph, w in s.modos:
            a = np.sin(P @ k + ph + w * t)
            u += np.outer(a, np.cross(k, d) / (np.linalg.norm(np.cross(k, d)) + 1e-9))
        return u * s.REMOLINO / np.sqrt(len(s.modos))

    def avanzar(s, t, dt=1 / 120):
        while s.t < t - 1e-9:
            h = min(dt, t - s.t)
            vivo = (s.nace <= s.t)
            if vivo.any():
                P, V = s.P[vivo], s.V[vivo]
                V += (-V / s.TAU + np.array([0, 0, s.SUBE / s.TAU]) + s._remolino(P, s.t) / s.TAU) * h
                P += V * h
                s.P[vivo], s.V[vivo] = P, V
            s.t += h
        # se borran las que ya se apagaron
        viva = (s.t - s.nace) < s.vida
        for nm in ('P', 'V', 'nace', 'vida', 'r0', 'r1', 'a0', 'tex_i', 'giro', 'tono'):
            setattr(s, nm, getattr(s, nm)[viva])

    def dibujar(s, r, hdr, camara, luz=1.0):
        """Mezcla el humo sobre la imagen HDR de r.draw (usa r.VP y la profundidad de r)."""
        if len(s.P) == 0:
            return hdr
        H, W = hdr.shape[:2]
        prof = r.lin_depth(0.02, 60)                       # metros, misma orientación que la imagen
        edad = s.t - s.nace
        ok = (edad >= 0) & (edad < s.vida)
        if not ok.any():
            return hdr
        P = s.P[ok]; edad = edad[ok]; vida = s.vida[ok]
        rad = s.r0[ok] + (s.r1[ok] - s.r0[ok]) * (1 - np.exp(-edad / 0.35))
        u = edad / vida
        alfa = s.a0[ok] * np.clip(edad / 0.08, 0, 1) * (1 - np.clip((u - 0.35) / 0.65, 0, 1)) ** 1.4
        c = (r.VP @ np.c_[P, np.ones(len(P))].T).T
        w = c[:, 3]; vis = w > 0.05
        x = (c[:, 0] / w * 0.5 + 0.5) * W; y = (1 - (c[:, 1] / w * 0.5 + 0.5)) * H
        f = r.Pm[1, 1] * H / 2                              # pixeles por metro a 1 m
        rp = rad * f / np.maximum(w, 1e-3)
        orden = np.argsort(-w)                             # de atrás hacia adelante
        out = hdr.copy()
        col = np.array(s.COLOR, np.float32) * luz
        tex_i = s.tex_i[ok]; giro = s.giro[ok]; tono = s.tono[ok]
        for i in orden:
            if not vis[i] or alfa[i] < 0.01 or rp[i] < 1.5:
                continue
            R = int(np.ceil(rp[i])); x0, y0 = int(x[i]) - R, int(y[i]) - R; x1, y1 = x0 + 2 * R, y0 + 2 * R
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                continue
            T = s.tex[tex_i[i]]
            M = cv2.getRotationMatrix2D((s.TEX / 2, s.TEX / 2), float(giro[i] + edad[i] * 40), 1.0)
            T = cv2.warpAffine(T, M, (s.TEX, s.TEX), flags=cv2.INTER_LINEAR, borderValue=0)
            T = cv2.resize(T, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
            G = cv2.resize(s.grad, (1, 2 * R))[:, :1]
            ax0, ay0 = max(0, -x0), max(0, -y0); bx0, by0 = max(0, x0), max(0, y0)
            bx1, by1 = min(W, x1), min(H, y1)
            a = T[ay0:ay0 + by1 - by0, ax0:ax0 + bx1 - bx0] * alfa[i]
            g = G[ay0:ay0 + by1 - by0]
            # lo que está delante tapa el humo; al tocar una superficie se funde suave
            d = prof[by0:by1, bx0:bx1]
            a = a * np.clip((d - w[i]) / (rad[i] * 1.2) + 0.5, 0, 1)
            reg = out[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1] = reg * (1 - a[..., None]) + (col[None, None, :] * (g[..., None] * tono[i])) * a[..., None]
        return out
