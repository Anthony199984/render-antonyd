# render-antonyd

Render de los videos de Antonydfilms repartido en hasta 20 máquinas de GitHub Actions (gratis en repositorio público).
- `hc/` = lo mínimo del motor que necesita el render (se enlaza como /home/claude en cada máquina).
- `trabajos/ACTUAL.json` = el trabajo (lo arma `herramientas/armar_trabajo.py plan.json 20`). Al subirlo, arranca solo.
- Cada máquina hace su trozo (`herramientas/trozo.py`); al final `herramientas/unir.py` arma el video y lo guarda en la rama `resultados`.
Repositorio temporal: se borra completo cuando se termine de usar.

## Max nuevo (13 nov 2026): cualquier video en 20 máquinas, con el proyecto CIFRADO
1. El video tiene un archivo con `preparar(ancho, alto) -> (cuadros, fps)` y `cuadro(k) -> imagen` (ejemplo: `max_nuevo/render_github_lovely.py`).
2. `python3 herramientas/mn_proyecto.py /home/claude/<video>.py 1080 1920` -> mide qué archivos usa, los cifra (AES-256) y los sube a la rama `proyecto-mn`.
3. `python3 herramientas/mn_plan.py NOMBRE 20 [--de K0 --a K1]` -> `trabajos/MN.json`; al subirlo, arranca `.github/workflows/max_nuevo.yml`.
4. El video sale en la rama `resultados/NOMBRE/` (con INFORME.md). Se baja con raw.githubusercontent.com.
La clave está en el secreto `CLAVE_PROYECTO` del repo (y en /home/claude/.clave_render_github del chat). Sin ella el proyecto no se abre.
