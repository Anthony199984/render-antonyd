# ACTUALIZACIÓN SOMA — Max sobre el esqueleto de Kimodo (27 oct 2026)

**Qué es:** el motor reproduce los BVH de Kimodo **tal cual salen**, sin retarget y sin hornear en Blender.
Max (la misma malla de `Personajes/Max/max_v1.glb`) se mueve con el esqueleto SOMA77 de Kimodo.

**No reemplaza nada:** max_v1, `retarget.py`, `retarget_max.py` y todos los videos anteriores siguen igual.
Cada video elige: Max clásico (mixamorig) o Max SOMA.

## Uso
```python
import sys; sys.path.insert(0, '/home/claude')
from soma import MaxSoma, Clip
m = MaxSoma()                          # carga Personajes/Max/max_v1.glb (1,874 m)
c = Clip('max_saludo.bvh')             # BVH exportado de Kimodo (demo web), sin tocar
res = m.pose(c, t)                     # [(V, N) por primitiva], Z arriba, igual que gltf.Skinned.pose
m.joint_pos('RightHand')               # posición de un hueso SOMA tras el pose (para IK, cámara, agarres)
m.set_morph({'mouthSmileLeft': .6})    # expresiones: blendshapes de Max, igual que antes
```
- `m.pose(c, t, bucle=True)` repite el clip. `m.globales(c, t, pies=False)` desactiva el bloqueo de pies.
- Los clips van a 30 fps; el motor interpola (slerp) para renderizar a 60 fps.
- Dibujar: igual que `Prota.draw` (`RS.set_mesh` + `skin.interleave` por primitiva). Ejemplo completo en `pruebas/video_soma.py`.

## Cómo funciona (por qué no hay retarget)
1. Las articulaciones SOMA se colocan dentro del cuerpo de Max (sus huesos mixamorig; los extra de SOMA en puntos
   proporcionales: metacarpos, Neck2, Spine1, puntas). Los largos de hueso son los de Max.
2. Max está en pose A. Para cada hueso se calcula **una sola vez** el giro que lleva el reposo del BVH a la pose de Max;
   ese giro es el bind. El giro global de Kimodo se aplica directo: `piel = [G_kimodo | p] · inv([A | pA])`.
   Probado: en reposo Max sale idéntico a max_v1 (error 0,0008 mm).
3. El reposo se lee de cada BVH (Kimodo por defecto exporta el reposo BONES-SEED, no la T-pose): funciona con los dos.
4. Pesos de piel: los de max_v1 pasados por nombre (tabla `PESOS` en `rig.py`); Neck se reparte entre Neck1 y Neck2.
5. Avance de la cadera escalado por el largo de pierna de Max (16 % más larga que la de Kimodo).
6. Bloqueo de pies: el apoyo se detecta en el propio esqueleto de Kimodo y el pie se clava con IK de dos huesos
   (el giro del pie es el de Kimodo). Medido: talón apoyado 0,5–0,7 cm/s (Kimodo original 0,2–0,4).

## Pipeline de animaciones con SOMA
- Kimodo (demo web https://nvidia-kimodo.hf.space, modelo **SOMA**, no SMPL-X) → exportar **BVH 30 fps** → al motor.
- **Ya no hace falta** pasar el clip por `retarget_max.py` ni por Blender para Max SOMA.
- Nombres de archivo: `max_<accion>.bvh` en la carpeta del video.

## Archivos
| Archivo | Qué es |
|---|---|
| `bvh.py` | Lector de BVH (cualquier BVH; Kimodo: Root → Hips → ZYX) con FK e interpolación |
| `rig.py` | `MaxSoma`: bind, pesos, pose, bloqueo de pies, `joint_pos` |
| `somaskel77_standard_tpose.bvh` | Esqueleto SOMA77 oficial de Kimodo (Apache 2.0) |
| `npz_a_bvh.py` | Convierte `motion.npz` de Kimodo a BVH (solo pruebas) |
| `pruebas_kimodo/*.bvh` | 8 movimientos reales de Kimodo-SOMA (ejemplos del repositorio de NVIDIA) |
| `pruebas/hoja_soma.py`, `pruebas/video_soma.py` | Hoja de poses y video de prueba |
| `LICENSE_kimodo_Apache2.txt` | Licencia del esqueleto y los ejemplos (Apache 2.0, no pide crédito en el video) |

7. Piel tapada: la piel del cuerpo bajo el pantalón y los tenis no se dibuja (asomaba por el pantalón al subir la pierna).

## Pendiente
- **Pesos del pantalón de max_v1** (viene del modelo, pasa igual con su esqueleto mixamorig): con zancadas largas o la pierna muy
  atrás, la zona de atrás del muslo se estira y se ve gris. Pasar peso a la cadera por código abrió la cintura (`CADERA_PANTALON = 0`).
  Arreglo de verdad: repintar pesos del pantalón (entrepierna/glúteos) en Blender, una vez, y sirve para max_v1 y Max SOMA.
- Dedos: los BVH de la demo web traen los dedos de Kimodo; los ejemplos `.npz` no (salen rectos).
- Comparación real max_v1 (retarget) vs Max SOMA con la misma animación: cuando llegue un BVH de Kimodo.
