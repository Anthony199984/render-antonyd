"""ZIP 40 · POLVO AL CAER: cuando el cuerpo blando de Max (ZIP 38) choca contra el piso levanta una nube baja de polvo.

Sale de cada parte del cuerpo justo cuando toca el piso, en cantidad según lo fuerte que pega (la cadera y el tronco
levantan más que una mano), se abre hacia afuera pegada al piso, sube muy poco y se asienta en ~1,5 s. Mismo sistema
de nubes que el humo (soma/humo.py, ZIP 39), con luz del sol (soma/humo_luz.py), color de tierra y nunca baja del piso.

    from soma.polvo import Polvo
    pv = Polvo(cb)                      # cb = CuerpoBlando(m, clip, t0) del ZIP 38
    # cada cuadro, en orden:
    pv.emitir_hasta(t); pv.avanzar(t)
    hdr = pv.dibujar(r, hdr, camara, sol=L['sun'])   # después de r.draw (y del humo), antes de hq.develop
Ajustes: Polvo.COLOR (color de la tierra), CANTIDAD (1 = lo aprobado), ALTO_TOCA (m).
"""
import numpy as np

from .humo_luz import HumoConLuz


class Polvo(HumoConLuz):
    COLOR = (0.52, 0.44, 0.34)       # tierra del camino (cambiar al color del piso de la escena)
    LUZ_ARRIBA = 0.45
    TAU = 0.45                       # el polvo pesa más que el humo: frena más despacio y se queda abajo
    SUBE = 0.05
    REMOLINO = 0.30
    CANTIDAD = 1.0
    ALTO_TOCA = 0.045                # m: una parte del cuerpo "toca" el piso por debajo de esta altura
    PISO = 0.0

    def __init__(s, cb, semilla=19):
        super().__init__(semilla)
        s.cb = cb
        s._impactos = s._buscar_impactos()
        s._hecho = -1

    def _buscar_impactos(s):
        """[(segundo, puntos, fuerza)] por cuadro de la simulación: dónde y cuán fuerte pega cada parte."""
        X = np.asarray(s.cb.frames); fps = s.cb.FPS; t0 = s.cb.t0
        z = X[:, :, 2] - s.PISO
        out = []
        abajo = z < s.ALTO_TOCA
        for f in range(1, len(X)):
            nuevo = abajo[f] & ~abajo[f - 1]
            if not nuevo.any():
                continue
            vz = (X[f - 1, nuevo, 2] - X[f, nuevo, 2]) * fps          # velocidad hacia abajo al tocar
            vh = (X[f, nuevo] - X[f - 1, nuevo]) * fps; vh[:, 2] = 0
            fuerza = np.clip(vz, 0, 4) + 0.3 * np.linalg.norm(vh, axis=1)
            fuerte = fuerza > 0.4
            if fuerte.any():
                out.append((t0 + f / fps, X[f, nuevo][fuerte], fuerza[fuerte], vh[fuerte]))
        s.centro = X[-1, :, :2].mean(0)
        return out

    def emitir_hasta(s, t):
        for k in range(s._hecho + 1, len(s._impactos)):
            ti, P, F, VH = s._impactos[k]
            if ti > t:
                break
            s._hecho = k
            n = int(min(140, 3.0 * F.sum() * s.CANTIDAD))
            if n <= 0:
                continue
            w = F / F.sum()
            idx = s.rng.choice(len(P), n, p=w)
            p = P[idx].copy(); p[:, 2] = np.maximum(p[:, 2], 0.02)
            d = p[:, :2] - s.centro; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            sale = np.c_[d * s.rng.uniform(0.5, 1.3, (n, 1)) * np.clip(F[idx], 0.5, 2.5)[:, None] * 0.6,
                         s.rng.uniform(0.05, 0.25, n)]
            vel = sale + VH[idx] * 0.4
            s.P = np.r_[s.P, p]; s.V = np.r_[s.V, vel]
            s.nace = np.r_[s.nace, np.full(n, ti)]
            U = lambda a, b: s.rng.uniform(a, b, n)
            s.vida = np.r_[s.vida, U(1.1, 1.9)]; s.r0 = np.r_[s.r0, U(0.04, 0.08)]; s.r1 = np.r_[s.r1, U(0.20, 0.42)]
            s.a0 = np.r_[s.a0, U(0.07, 0.16)]; s.tex_i = np.r_[s.tex_i, s.rng.integers(0, len(s.tex), n)]
            s.giro = np.r_[s.giro, U(0, 360)]; s.tono = np.r_[s.tono, U(0.82, 1.05)]

    def avanzar(s, t, dt=1 / 120):
        super().avanzar(t, dt)
        if len(s.P):
            s.P[:, 2] = np.maximum(s.P[:, 2], s.PISO + 0.02)       # el polvo nunca baja del piso
