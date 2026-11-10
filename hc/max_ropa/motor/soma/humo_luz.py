"""ZIP 40 · HUMO Y POLVO CON LUZ DEL SOL (archivo nuevo: humo.py del ZIP 39 queda igual).

Antes cada nube era del mismo gris por todos lados (solo un poco más clara arriba). Ahora:
  - el LADO DEL SOL de cada nube es más claro y cálido, y el lado contrario más gris y frío;
  - las nubes que tienen otras delante hacia el sol quedan en SOMBRA (el centro de una nube densa es más oscuro que
    sus bordes iluminados): así se ve el volumen y no una mancha plana.

    from soma.humo_luz import HumoConLuz
    h = HumoConLuz()                         # igual que Humo (mismas funciones)
    hdr = h.dibujar(r, hdr, camara, sol=L['sun'])     # sol = la dirección del sol de la escena (la misma de hq)
El polvo (soma/polvo.py) ya la usa. Se apaga con HumoConLuz.LUZ_SOL = False.
"""
import numpy as np
import cv2

from .humo import Humo


class HumoConLuz(Humo):
    LUZ_SOL = True
    AMBIENTE = 0.55            # luz que llega a la sombra (cielo)
    SOL = 0.62                 # luz directa del sol en el lado iluminado
    SOMBRA_K = 0.9             # cuánto oscurece la nube que tiene otras delante hacia el sol
    LADO_K = 0.38              # diferencia entre el lado del sol y el contrario dentro de cada nube
    TIBIO = (1.06, 1.0, 0.92)  # tinte del lado del sol
    FRIO = (0.90, 0.95, 1.05)  # tinte de la sombra

    def _sombras(s, P, rad, alfa, sol):
        from scipy.spatial import cKDTree
        if len(P) < 2:
            return np.ones(len(P))
        T = cKDTree(P)
        dens = np.zeros(len(P))
        for d in (0.6, 1.2, 2.0):
            Q = P + sol[None, :] * rad[:, None] * d
            for i, vec in enumerate(T.query_ball_point(Q, r=rad * 0.9)):
                vec = [k for k in vec if k != i]
                if vec:
                    dens[i] += alfa[vec].sum()
        return np.exp(-s.SOMBRA_K * dens)

    def dibujar(s, r, hdr, camara, luz=1.0, sol=(-0.35, -0.45, 0.82)):
        if not s.LUZ_SOL:
            return super().dibujar(r, hdr, camara, luz)
        if len(s.P) == 0:
            return hdr
        H, W = hdr.shape[:2]
        sol = np.asarray(sol, float); sol /= np.linalg.norm(sol)
        prof = r.lin_depth(0.02, 60)
        edad = s.t - s.nace
        ok = (edad >= 0) & (edad < s.vida)
        if not ok.any():
            return hdr
        P = s.P[ok]; edad = edad[ok]; vida = s.vida[ok]
        rad = s.r0[ok] + (s.r1[ok] - s.r0[ok]) * (1 - np.exp(-edad / 0.35))
        u = edad / vida
        alfa = s.a0[ok] * np.clip(edad / 0.08, 0, 1) * (1 - np.clip((u - 0.35) / 0.65, 0, 1)) ** 1.4
        sombra = s._sombras(P, rad, alfa, sol)
        def proy(X):
            c = (r.VP @ np.c_[X, np.ones(len(X))].T).T
            return c, (c[:, 0] / c[:, 3] * 0.5 + 0.5) * W, (1 - (c[:, 1] / c[:, 3] * 0.5 + 0.5)) * H
        c, x, y = proy(P)
        _, xs, ys = proy(P + sol[None, :] * rad[:, None])
        w = c[:, 3]; vis = w > 0.05
        f = r.Pm[1, 1] * H / 2
        rp = rad * f / np.maximum(w, 1e-3)
        orden = np.argsort(-w)
        out = hdr.copy()
        base = np.array(s.COLOR, np.float32) * luz
        tib = np.array(s.TIBIO, np.float32); fri = np.array(s.FRIO, np.float32)
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
            # hacia dónde queda el sol en pantalla, dentro de esta nube
            dsx, dsy = xs[i] - x[i], ys[i] - y[i]; n2 = np.hypot(dsx, dsy)
            yy, xx = np.mgrid[-1:1:complex(2 * R), -1:1:complex(2 * R)]
            lado = (xx * dsx + yy * dsy) / n2 if n2 > 1e-3 else -yy
            ax0, ay0 = max(0, -x0), max(0, -y0); bx0, by0 = max(0, x0), max(0, y0)
            bx1, by1 = min(W, x1), min(H, y1)
            sl = (slice(ay0, ay0 + by1 - by0), slice(ax0, ax0 + bx1 - bx0))
            a = T[sl] * alfa[i]
            k = np.clip(0.5 + 0.5 * lado[sl] * s.LADO_K * 2.0, 0, 1)            # 1 = lado del sol
            luzp = s.AMBIENTE + s.SOL * sombra[i] * (0.35 + 0.65 * k)
            tinte = fri[None, None, :] * (1 - k[..., None]) + tib[None, None, :] * k[..., None]
            col = base[None, None, :] * tinte * (luzp * tono[i])[..., None]
            d = prof[by0:by1, bx0:bx1]
            a = a * np.clip((d - w[i]) / (rad[i] * 1.2) + 0.5, 0, 1)
            reg = out[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1] = reg * (1 - a[..., None]) + col * a[..., None]
        return out
