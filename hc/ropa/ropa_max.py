"""ROPA DE MAX ARREGLADA (capas nuevas; no cambia archivos del motor).

    from ropa_max import max_arreglado
    from soma.max_v17 import MaxCaraV17          # o la versión de Max que use el motor
    Max = max_arreglado(MaxCaraV17)              # la misma clase + los arreglos de abajo
    m = Max(glb=...)

Arreglos (cada uno se apaga con su bandera = False):
 1. MANGA_ACOTADA: la capa "manga limpia" del motor movía, por error, CUALQUIER punto de la camiseta que quedara en
    línea con el borde de la manga (un tubo sin fin hacia el codo). Con el brazo colgando pegado al cuerpo, ese tubo
    atravesaba el costado y el bajo de la camiseta y los subía hasta la manga: se veían tiras y capas rotas debajo del
    brazo. Ahora solo mueve puntos de la manga y solo los que están a pocos centímetros del borde.
 2. PANTALON_PEGADO (pantalon_pegado.py): al agacharse, cada pierna del pantalón va con su pierna y no se infla.
"""
import numpy as np
from pantalon_pegado import _PantalonPegado
from cuerpo_solido import _CuerpoSolido
from camiseta_planchada import _CamisetaPlanchada


class _MangaAcotada:
    MANGA_ACOTADA = True
    MANGA_TUBO = 0.05      # m: hasta dónde, pasado el borde, se recoge la tela de la manga

    def _detras_del_borde(s, d, X, lazos, atras):
        if not s.MANGA_ACOTADA:
            return super()._detras_del_borde(d, X, lazos, atras)
        k = next(k_ for k_, v in s._mtela.items() if v is d)
        manga = s._prep_axila(k)['manga']
        en_borde = np.zeros(len(X), bool)
        for _, lz in lazos: en_borde[lz] = True
        for L, (Y, rad, nrm, cY, rmax) in s._borde_manga.items():
            u = (X - cY) @ nrm
            ra = np.linalg.norm((X - cY) - np.outer(u, nrm), axis=1)
            sel = (~en_borde) & (manga[L] > 0.5) & (ra < rmax + 0.02) & (u > -atras) & (u < s.MANGA_TUBO)
            if sel.any():
                X[sel] -= np.outer(u[sel] + atras, nrm)
        return X


CAPAS = (_CuerpoSolido, _CamisetaPlanchada, _MangaAcotada, _PantalonPegado)


def max_arreglado(Clase):
    """La misma clase de Max con la ropa arreglada encima (no modifica la clase original)."""
    return type(Clase.__name__ + 'Ropa', CAPAS + (Clase,), {})
