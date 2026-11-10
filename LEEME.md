# render-antonyd

Render de los videos de Antonydfilms repartido en hasta 20 máquinas de GitHub Actions (gratis en repositorio público).
- `hc/` = lo mínimo del motor que necesita el render (se enlaza como /home/claude en cada máquina).
- `trabajos/ACTUAL.json` = el trabajo (lo arma `herramientas/armar_trabajo.py plan.json 20`). Al subirlo, arranca solo.
- Cada máquina hace su trozo (`herramientas/trozo.py`); al final `herramientas/unir.py` arma el video y lo guarda en la rama `resultados`.
Repositorio temporal: se borra completo cuando se termine de usar.
