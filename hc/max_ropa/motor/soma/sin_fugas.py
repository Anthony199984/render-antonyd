"""ZIP 40 · RENDERS LARGOS SIN QUEDARSE SIN MEMORIA.

El problema: en cada cuadro los scripts vuelven a subir las mallas de Max con r.add_mesh(...) (antes borrando el VAO).
El VAO se borraba pero la MEMORIA DE LA MALLA (el VBO) no: cada cuadro dejaba guardada la malla vieja. En un render
largo (60 FPS, 1080x1920, un minuto = 3 600 cuadros) la memoria se llena y el render se corta solo a mitad de camino.

Arreglo (no cambia ningún archivo del motor): cuando se sube una malla con un nombre que ya existe, se REUSA su memoria
(o, si el script borró el VAO, se libera la vieja antes de crear la nueva). Se activa de dos formas:

  1) En un script nuevo, una línea después de importar hq:
         import hq
         import soma.sin_fugas            # (se activa sola al importarla)
  2) Para un script que ya existe (revision_max.py, sin_esqueleto.py, el render de un video...), sin tocarlo:
         python render_sin_fugas.py revision_max.py <los mismos argumentos de siempre>
"""
import ctypes as C
import os
import sys

import numpy as np

_ARRAY_BUFFER, _BINDING = 0x8892, 0x8894


def _parchar(HQ):
    if getattr(HQ, '_sin_fugas', False):
        return
    original = HQ.add_mesh

    def add_mesh(self, name, data):
        reg = self.__dict__.setdefault('_vbos', {})
        g = self.g
        if name in reg:
            vao, vbo = reg[name]
            g.glIsVertexArray.argtypes = [C.c_uint]; g.glIsVertexArray.restype = C.c_ubyte
            if self.meshes.get(name, (None,))[0] == vao and g.glIsVertexArray(vao):
                # la misma malla con datos nuevos: se reusa su memoria
                data = np.ascontiguousarray(data, np.float32)
                self.glBindVertexArray(vao); self.glBindBuffer(_ARRAY_BUFFER, vbo)
                self.glBufferData(_ARRAY_BUFFER, data.nbytes, data.ctypes.data, 0x88E4)
                self.meshes[name] = (vao, len(data))
                if hasattr(self, 'mesh_data'):
                    self.mesh_data[name] = data
                return
            # el script borró el VAO: se libera también la memoria de la malla vieja
            g.glDeleteBuffers.argtypes = [C.c_int, C.POINTER(C.c_uint)]
            g.glDeleteBuffers(1, C.byref(C.c_uint(vbo)))
        original(self, name, data)
        g.glGetIntegerv.argtypes = [C.c_uint, C.POINTER(C.c_int)]
        v = C.c_int(); g.glGetIntegerv(_BINDING, C.byref(v))
        reg[name] = (self.meshes[name][0], v.value)

    HQ.add_mesh = add_mesh
    HQ._sin_fugas = True


def activar():
    """Arregla hq.HQ (y gl_engine.Renderer si ya está cargado)."""
    hechos = 0
    for nombre in ('hq', 'gl_engine'):
        mod = sys.modules.get(nombre)
        if mod is None:
            continue
        for cls in ('HQ', 'Renderer'):
            if hasattr(mod, cls):
                _parchar(getattr(mod, cls)); hechos += 1
    return hechos


activar()
