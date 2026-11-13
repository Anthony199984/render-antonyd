#!/bin/bash
# Espera el resultado de un render en GitHub y lo baja (sin git fetch de la rama resultados, que llena el disco).
#   bash herramientas/mn_bajar.sh NOMBRE carpeta_destino [minutos_maximos]
N=$1; D=$2; MAX=${3:-60}; R=Anthony199984/render-antonyd; mkdir -p "$D"
for i in $(seq 1 $MAX); do
  if gh api "repos/$R/contents/$N?ref=resultados" --jq '.[].name' > /tmp/mn_lista.txt 2>/dev/null && grep -q "SHA256.txt" /tmp/mn_lista.txt; then
    for f in $(cat /tmp/mn_lista.txt); do gh api -H "Accept: application/vnd.github.raw" "repos/$R/contents/$N/$f?ref=resultados" > "$D/$f"; done
    for f in "$D"/*.mp4.parte00; do [ -f "$f" ] && cat "${f%00}"* > "${f%.parte00}"; done
    echo "BAJADO $N en $D:"; ls -la "$D"; exit 0
  fi
  sleep 60
done
echo "todavía no hay resultado de $N después de $MAX minutos"; exit 1
