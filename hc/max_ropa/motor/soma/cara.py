"""Cara viva para Max (ZIP_26, archivo NUEVO: no cambia nada de lo que ya hay).

Max tiene en la cara las 52 formas ARKit (cejas, ojos, boca, mejillas…) más 8 visemas. Hasta ahora casi no se usaban:
los gestos de la biblioteca traían la expresión solo escrita en el catálogo, y la mirada (mirada.py) giraba los ojos
del esqueleto de Kimodo, pero Max no tiene huesos de ojos: sus ojos se movían solo en la vista previa con maniquí.

Este módulo hace la cara sola, cuadro a cuadro:
  1. EXPRESIÓN por acción: cada gesto de la biblioteca tiene su cara (tabla GESTO_CARA, sacada de "cara_sugerida"
     del catálogo). En el JSON de la escena se puede poner "cara": "sorpresa" en cualquier acción para cambiarla.
     Una sola expresión por acción, con entrada y salida suaves (no salta de una cara a otra).
  2. OJOS que miran de verdad: el giro de LeftEye/RightEye que pone mirada.py (o Kimodo) se pasa a las formas
     eyeLookIn/Out/Up/Down de Max (26° a los lados, 20° arriba/abajo, medidos en su modelo).
  3. PARPADEO natural: cada 2,5–5,5 s al azar (repetible con semilla), más rápido al cerrar que al abrir, y un
     parpadeo extra en los giros grandes de cabeza; no parpadea si la expresión ya tiene los ojos cerrados.
  4. VIDA: micro-movimiento lento de cejas y boca (muy suave) para que la cara nunca quede de estatua, y boca
     "hablando" en los gestos de explicar (sin voz: no es sincronía de labios).
  5. AGARRES: cara concentrada al ir a agarrar y mientras sostiene (sale del informe de la escena).

Uso en el render (en vez de MaxSoma / MaxV3 / MaxV4):
    from soma.cara import MaxCara, PistaCara
    max = MaxCara(glb='/home/claude/Personajes/Max/max_v4.glb')        # igual que MaxV4 (camiseta=False, etc.)
    max.pista = PistaCara.desde_escena('motionbricks_cpu/salidas/mi_escena')   # carpeta de salida de escena.py
    res = max.pose(clip, t)                                              # la cara ya va puesta
    # o a mano:  max.pista = PistaCara([(0.0, 2.0, 'sonrisa'), (2.0, 4.5, 'sorpresa')])
Sin pista, MaxCara igual parpadea y mueve los ojos según la mirada.
"""
import json
from pathlib import Path

import numpy as np

from .max_v4 import MaxV4

# ------------------------------------------------------------------ expresiones (formas ARKit de Max, 0..1)
def _par(nombre, v):
    return {nombre + "Left": v, nombre + "Right": v}


def _m(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = max(out.get(k, 0.0), v)
    return out


PRESETS = {
    # (probadas en la cara de Max con render; fuertes a propósito: en un Short la cara se ve chica y tiene que leerse)
    "neutral": {},
    "atento": _m({"browInnerUp": 0.15}, _par("eyeWide", 0.12)),
    "concentrado": _m(_par("browDown", 0.40), _par("eyeSquint", 0.25), _par("mouthPress", 0.40)),
    "sonrisa": _m(_par("mouthSmile", 1.0), _par("cheekSquint", 0.60), _par("eyeSquint", 0.35), _par("mouthDimple", 0.3),
                  _par("mouthUpperUp", 0.15), {"jawOpen": 0.05}),
    "simpatico": _m(_par("mouthSmile", 0.85), _par("cheekSquint", 0.45), _par("eyeSquint", 0.25), {"browInnerUp": 0.25}),
    "contento": _m(_par("mouthSmile", 1.0), {"jawOpen": 0.22}, _par("cheekSquint", 0.65), _par("eyeSquint", 0.40),
                   _par("mouthUpperUp", 0.30)),
    "euforia": _m(_par("mouthSmile", 1.0), {"jawOpen": 0.40}, _par("cheekSquint", 0.70), _par("eyeSquint", 0.60),
                  _par("eyeBlink", 0.45), _par("browOuterUp", 0.40), _par("mouthUpperUp", 0.20)),
    "de_acuerdo": _m(_par("mouthSmile", 0.80), _par("eyeSquint", 0.45), _par("cheekSquint", 0.45), {"browInnerUp": 0.2}),
    "sorpresa": _m(_par("eyeWide", 1.0), {"browInnerUp": 0.90}, _par("browOuterUp", 0.90), {"jawOpen": 0.45},
                   {"mouthFunnel": 0.35}),
    "shock": _m(_par("eyeWide", 1.0), {"browInnerUp": 1.0}, _par("browOuterUp", 0.70), {"jawOpen": 0.60},
                _par("mouthStretch", 0.35)),
    "susto": _m(_par("eyeWide", 1.0), {"browInnerUp": 1.0}, _par("browOuterUp", 0.55), {"jawOpen": 0.40},
                _par("mouthStretch", 0.75), _par("mouthLowerDown", 0.35)),
    "confundido": {"browDownLeft": 0.90, "browOuterUpRight": 1.0, "browInnerUp": 0.30, "mouthLeft": 0.60,
                   "mouthPressRight": 0.50, "eyeSquintLeft": 0.50, "mouthRollLower": 0.20, "cheekSquintLeft": 0.25},
    "pensativo": _m({"mouthPucker": 0.30, "mouthRollLower": 0.50}, _par("mouthPress", 0.60),
                    {"browDownLeft": 0.50, "browOuterUpRight": 0.45, "browInnerUp": 0.20},
                    _par("eyeLookUp", 0.50), {"eyeLookOutRight": 0.40, "eyeLookInLeft": 0.40}, {"eyeSquintLeft": 0.25}),
    "no_se": _m({"browInnerUp": 0.90}, _par("browOuterUp", 0.90), _par("mouthFrown", 1.0),
                {"mouthShrugLower": 0.90, "mouthRollLower": 0.30}, _par("mouthPress", 0.40)),
    "desaprobacion": _m(_par("browDown", 0.90), _par("mouthPress", 0.80), _par("mouthFrown", 0.60),
                        _par("noseSneer", 0.35), {"mouthShrugLower": 0.30}, _par("eyeSquint", 0.25)),
    "decidido": _m(_par("browDown", 0.50), _par("eyeSquint", 0.30), _par("mouthPress", 0.30), {"jawForward": 0.15}),
    "frustracion": _m(_par("eyeBlink", 0.90), _par("browDown", 0.90), _par("mouthPress", 0.60),
                      _par("mouthFrown", 0.60), _par("noseSneer", 0.40), {"mouthShrugLower": 0.3}),
    "nervioso": _m({"browInnerUp": 1.0}, _par("mouthStretch", 0.70), _par("mouthPress", 0.30), _par("eyeWide", 0.50),
                   _par("mouthFrown", 0.30)),
    "impaciente": _m(_par("browDown", 0.70), {"mouthLeft": 0.70, "mouthPressLeft": 0.60, "cheekSquintLeft": 0.30},
                     _par("eyeSquint", 0.30)),
    "explicando": _m({"browInnerUp": 0.50}, _par("browOuterUp", 0.40), _par("mouthSmile", 0.30)),
    "curioso": _m({"browInnerUp": 0.60, "browOuterUpLeft": 0.70, "eyeSquintRight": 0.40, "mouthPucker": 0.30},
                  _par("eyeWide", 0.15)),
    "cansado": _m(_par("eyeBlink", 0.50), _par("eyeSquint", 0.30), {"jawOpen": 0.32}, {"mouthFunnel": 0.25},
                  {"browInnerUp": 0.45}, _par("mouthFrown", 0.30)),
    "esfuerzo": _m(_par("browDown", 0.85), _par("eyeSquint", 0.80), _par("mouthStretch", 0.90), _par("mouthPress", 0.30),
                   _par("noseSneer", 0.50), {"jawForward": 0.20}, _par("cheekSquint", 0.50), _par("mouthUpperUp", 0.35),
                   _par("mouthLowerDown", 0.30)),
    "dolor": _m(_par("eyeBlink", 0.75), _par("browDown", 0.60), {"browInnerUp": 0.60}, _par("mouthStretch", 0.90),
                _par("noseSneer", 0.60), {"jawOpen": 0.15}, _par("mouthUpperUp", 0.40), _par("mouthLowerDown", 0.30)),
    "asco": _m(_par("noseSneer", 1.0), _par("mouthUpperUp", 0.90), _par("browDown", 0.60), _par("mouthFrown", 0.50),
               _par("eyeSquint", 0.50), _par("cheekSquint", 0.40)),
    "triste": _m({"browInnerUp": 1.0}, _par("mouthFrown", 1.0), {"mouthShrugLower": 0.50}, _par("mouthPress", 0.30),
                 _par("eyeLookDown", 0.35), _par("eyeSquint", 0.20)),
    "enojo": _m(_par("browDown", 1.0), _par("eyeSquint", 0.60), _par("noseSneer", 0.70), _par("mouthPress", 0.60),
                _par("mouthFrown", 0.30), {"jawForward": 0.25}),
}
SINONIMOS = {"shock": "shock", "en_shock": "shock", "miedo": "susto", "alegria": "contento", "feliz": "contento",
             "sorprendido": "sorpresa", "enojado": "enojo", "rabia": "enojo", "asustado": "susto", "pensando": "pensativo",
             "triste": "triste", "dolor": "dolor", "aburrido": "cansado", "neutro": "neutral"}

# gesto de la biblioteca -> (expresión, hablar). Sale de "cara_sugerida" de biblioteca_gestos/catalogo.json
GESTO_CARA = {
    "rascarse_cabeza_confundido": ("confundido", False), "mano_barbilla_pensando": ("pensativo", False),
    "manos_a_la_cabeza": ("shock", False), "encogerse_hombros": ("no_se", False),
    "asustarse_salto_atras": ("susto", False), "asentir": ("de_acuerdo", False),
    "negar_con_cabeza": ("desaprobacion", False), "celebrar_punos_arriba": ("euforia", False),
    "senalar_adelante": ("decidido", False), "mano_en_la_cara": ("frustracion", False),
    "mirar_alrededor_nervioso": ("nervioso", False), "aplaudir": ("contento", False),
    "brazos_cruzados_impaciente": ("impaciente", False), "sorpresa_retroceder": ("sorpresa", False),
    "caminar": ("atento", False), "espera_respirando": ("neutral", False),
    "explicar_con_manos": ("explicando", True), "saludar": ("simpatico", False),
    "agacharse_mirar_piso": ("curioso", False), "limpiarse_sudor_frente": ("cansado", False),
}
OJO_NEUTRO = {"Left": (-0.18472, -0.06591, 0.00631), "Right": (-0.18497, 0.06572, -0.00622)}   # giro local de ojo
# que deja Kimodo cuando NO mueve los ojos en algunas salidas. Los BVH de escena.py y de la biblioteca traen los ojos
# en 0 (sin giro). Por eso el neutro se elige por clip: el que esté más cerca de lo que trae el clip (ver _neutro_ojos).
OJOS_LADO, OJOS_ARRIBA = 26.0, 20.0          # grados que da cada forma eyeLook a peso 1 (medido en max_v3)
ENTRADA_S, SALIDA_S = 0.30, 0.35             # transición de una cara a otra


def _rot(rv):
    rv = np.asarray(rv, float); a = np.linalg.norm(rv)
    if a < 1e-12: return np.eye(3)
    k = rv / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _rv(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def _suave(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def resolver(nombre):
    n = str(nombre).strip().lower().replace(" ", "_")
    n = SINONIMOS.get(n, n)
    if n not in PRESETS:
        raise KeyError(f"expresión desconocida: {nombre} (hay: {', '.join(sorted(PRESETS))})")
    return n


# ------------------------------------------------------------------ pista de cara
class PistaCara:
    """Línea de tiempo de expresiones: [(t0, t1, expresión, intensidad?, hablar?)] en segundos."""

    def __init__(s, tramos=(), semilla=7, parpadeo=True, vida=True, duracion=None):
        s.tramos = []
        for tr in tramos:
            t0, t1, ex = tr[:3]
            inten = tr[3] if len(tr) > 3 else 1.0
            hab = tr[4] if len(tr) > 4 else (ex == "explicando")
            s.tramos.append((float(t0), float(t1), resolver(ex), float(inten), bool(hab)))
        s.tramos.sort()
        s.semilla = semilla; s.parpadeo = parpadeo; s.vida = vida
        fin = max([t1 for _, t1, *_ in s.tramos], default=0.0)
        s.duracion = max(duracion or 0.0, fin) + 10.0
        rng = np.random.default_rng(semilla)
        t, s.parpadeos = rng.uniform(0.5, 2.0), []
        while t < s.duracion:
            s.parpadeos.append(t); t += rng.uniform(2.5, 5.5)
        s.parpadeos = np.array(s.parpadeos)
        s._fase = rng.uniform(0, 2 * np.pi, 8)

    # -------- construir desde la salida de escena.py
    @classmethod
    def desde_escena(cls, carpeta, escena_json=None, semilla=7):
        carpeta = Path(carpeta)
        npz = next(carpeta.glob("*_L77.npz"))
        d = np.load(npz, allow_pickle=True)
        tag = [str(x) for x in d["tag"]]
        fps = 30.0
        informe = {}
        inf_p = next(carpeta.glob("*_informe.json"), None)
        if inf_p is not None:
            informe = json.loads(inf_p.read_text(encoding="utf-8"))
        caras_json = {}
        esc = {}
        if escena_json is not None:
            esc = json.loads(Path(escena_json).read_text(encoding="utf-8"))
            for a in esc.get("acciones", []):
                if "cara" in a:
                    clave = a.get("nombre") or Path(a.get("gesto", a.get("kimodo", ""))).stem
                    clave = clave[4:] if clave[:1] == "g" and clave[1:3].isdigit() else clave
                    caras_json[clave.replace("_", " ")] = a["cara"]
        agarres_json = {}
        if escena_json is not None:
            for a in esc.get("acciones", []):
                if "agarre" in a:
                    agarres_json[a["agarre"].get("objeto")] = a["agarre"]
        # tramos: cada acción con su cara; la transición IA hacia una acción ya pone la cara de esa acción
        # (la cara se adelanta al cuerpo, como en una persona); acciones seguidas con la misma cara se unen
        tramos = []
        for f, nom in enumerate(tag):
            nom = nom.split("→")[-1].strip()
            ex, hab = cls._cara_de(nom, caras_json)
            if tramos and tramos[-1][2] == ex and tramos[-1][4] == hab:
                tramos[-1][1] = (f + 1) / fps
            else:
                tramos.append([f / fps, (f + 1) / fps, ex, 1.0, hab])
        tramos = [tuple(t) for t in tramos]
        # agarres: concentrado desde que sale la mano hasta que suelta (encima de la expresión de la acción)
        for a in informe.get("agarres", []):
            try:
                ts = float(a["segundo"])
            except (KeyError, TypeError, ValueError):
                continue
            g = agarres_json.get(a.get("objeto"), {})
            alc = a.get("alcanzar") or g.get("alcanzar") or 0.9
            sos = a.get("sostener") or g.get("sostener") or 2.0
            tramos.append((ts - float(alc), ts + float(sos), "concentrado", 0.8, False))
        p = cls(tramos, semilla=semilla, duracion=len(tag) / fps)
        p._capas = True
        return p

    @staticmethod
    def _cara_de(nombre_tag, caras_json):
        n = nombre_tag.strip()
        if n in caras_json:
            return resolver(caras_json[n]), resolver(caras_json[n]) == "explicando"
        clave = n.replace(" ", "_")
        for g, (ex, hab) in GESTO_CARA.items():
            if g in clave:
                return ex, hab
        if clave.startswith(("caminar", "camina", "agacharse")):
            return "atento", False
        return "neutral", False

    # -------- pesos en el tiempo t
    def pesos(s, t):
        w = {}
        capas = getattr(s, "_capas", False)
        for t0, t1, ex, inten, hab in s.tramos:
            if t < t0 - ENTRADA_S or t > t1 + SALIDA_S:
                continue
            a = _suave((t - (t0 - ENTRADA_S * 0.5)) / ENTRADA_S) * _suave(((t1 + SALIDA_S * 0.5) - t) / SALIDA_S) * inten
            if a <= 0 or (capas and ex == "concentrado"):
                continue
            for k, v in PRESETS[ex].items():
                w[k] = max(w.get(k, 0.0), v * a)
            if hab:
                s._hablar(w, t, a)
        if capas:                         # agarres: la cara se concentra encima de la expresión de la acción
            for t0, t1, ex, inten, hab in s.tramos:
                if ex != "concentrado" or t < t0 - ENTRADA_S or t > t1 + SALIDA_S:
                    continue
                a = _suave((t - (t0 - ENTRADA_S * 0.5)) / ENTRADA_S) * _suave(((t1 + SALIDA_S * 0.5) - t) / SALIDA_S) * inten
                for k in list(w):
                    w[k] *= (1 - 0.4 * a)
                for k, v in PRESETS["concentrado"].items():
                    w[k] = max(w.get(k, 0.0), v * a)
        if s.vida:
            f = s._fase
            w["browInnerUp"] = w.get("browInnerUp", 0) + 0.05 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.21 * t + f[0]))
            for lado, ph in (("Left", f[1]), ("Right", f[2])):
                k = "mouthSmile" + lado
                w[k] = w.get(k, 0) + 0.03 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * t + ph))
        if s.parpadeo:
            cerr = max(w.get("eyeBlinkLeft", 0), w.get("eyeBlinkRight", 0))
            bl = s._parpadeo(t) * (1 - cerr)
            for lado in ("Left", "Right"):
                k = "eyeBlink" + lado
                w[k] = min(1.0, w.get(k, 0) + bl)
        return {k: float(np.clip(v, 0, 1)) for k, v in w.items() if v > 1e-3}

    def _parpadeo(s, t, cerrar=0.07, quieto=0.03, abrir=0.13):
        if not len(s.parpadeos):
            return 0.0
        i = np.searchsorted(s.parpadeos, t) - 1
        if i < 0:
            return 0.0
        dt = t - s.parpadeos[i]
        if dt < cerrar: return _suave(dt / cerrar)
        if dt < cerrar + quieto: return 1.0
        if dt < cerrar + quieto + abrir: return 1.0 - _suave((dt - cerrar - quieto) / abrir)
        return 0.0

    def _hablar(s, w, t, a):
        f = s._fase
        sil = 0.5 + 0.5 * np.sin(2 * np.pi * 3.3 * t + f[3]) * np.sin(2 * np.pi * 0.9 * t + f[4])
        sil = max(0.0, sil) ** 1.5
        w["jawOpen"] = max(w.get("jawOpen", 0), 0.28 * sil * a)
        w["V_Open"] = max(w.get("V_Open", 0), 0.35 * sil * a)
        w["V_Wide"] = max(w.get("V_Wide", 0), 0.2 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t + f[5])) * a)

    def agregar_parpadeo(s, t):
        """Parpadeo extra en el segundo t (p. ej. al girar la cabeza o en un corte de cámara)."""
        s.parpadeos = np.sort(np.r_[s.parpadeos, float(t)])


# ------------------------------------------------------------------ Max con cara
class MaxCara(MaxV4):
    """MaxV4 + cara viva. Mismo uso que MaxV4. `pista` = PistaCara (o None: solo parpadeo y ojos)."""
    OJOS = True               # pasar el giro de los ojos de Kimodo/mirada.py a las formas eyeLook de Max
    PARPADEO_EN_GIRO = 55.0   # grados/s de giro de cabeza que provocan un parpadeo

    # cuello: cuando la animación agacha mucho la cabeza (mirar la mesa, agarrar algo abajo), Max la agacha menos y
    # baja más los ojos, como una persona. Los ojos siguen mirando al mismo punto (sus giros no se tocan).
    CUELLO_LIBRE = 25.0       # grados de cabeza agachada (respecto al pecho) que se dejan tal cual
    CUELLO_QUITA = 0.6        # del exceso sobre CUELLO_LIBRE, qué parte se quita
    CUELLO_MAX_QUITA = 18.0   # grados como máximo que se levanta (lo que los ojos pueden compensar)

    def __init__(s, *a, pista=None, **k):
        super().__init__(*a, **k)
        s.pista = pista
        s._sin_pista = PistaCara()
        s._ultimo = None
        s.cara_manual = {}

    def set_morph(s, weights):
        """Compatibilidad: si el código de la escena pone una cara a mano (m.set_morph({...}) o s.cara), esa cara se
        guarda y se SUMA a la automática en cada cuadro (gana el valor más alto de cada forma)."""
        s.cara_manual = dict(weights or {})

    def globales(s, clip, t, bucle=False, pies=True):
        G, Pm, b = super().globales(clip, t, bucle, pies)
        if s.CUELLO_QUITA > 0:
            s._limitar_cuello(G, Pm, b)
        return G, Pm, b

    def _limitar_cuello(s, G, Pm, b):
        ix = s.idx
        Kc = G[ix['Chest']] @ b['A'][ix['Chest']].T
        Kh = G[ix['Head']] @ b['A'][ix['Head']].T
        fh, fc, uc, xc = Kh @ [0, 0, 1.0], Kc @ [0, 0, 1.0], Kc @ [0, 1.0, 0], Kc @ [1.0, 0, 0]
        agachada = np.degrees(np.arctan2(-(fh @ uc), fh @ fc))
        if agachada <= s.CUELLO_LIBRE:
            s.cuello_levantado = 0.0
            return
        quita = min((agachada - s.CUELLO_LIBRE) * s.CUELLO_QUITA, s.CUELLO_MAX_QUITA)
        s.cuello_levantado = quita
        if not hasattr(s, '_sub_cuello'):
            s._sub_cuello = {}
            for n in ('Neck1', 'Neck2', 'Head'):
                d = [ix[n]]
                for j in range(len(s.nombres)):
                    if j not in d and s.padres[j] in d: d.append(j)
                s._sub_cuello[n] = sorted(d)
        ojos = {ix[n] for n in ('LeftEye', 'RightEye') if n in ix}
        g = np.radians(quita / 3.0)
        e = xc / np.linalg.norm(xc)
        Kx = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        for sg in (1.0, -1.0):                      # el sentido que levanta la cabeza
            R = np.eye(3) + np.sin(sg * g) * Kx + (1 - np.cos(sg * g)) * Kx @ Kx
            if ((R @ fh) @ uc) > fh @ uc: break
        for n in ('Neck1', 'Neck2', 'Head'):        # repartido en las 3 vértebras del cuello
            piv = Pm[ix[n]].copy()
            for j in s._sub_cuello[n]:
                Pm[j] = piv + R @ (Pm[j] - piv)
                if j not in ojos:
                    G[j] = R @ G[j]

    def _neutro_ojos(s, lado):
        """Giro de ojo que cuenta como "mirar al frente" para el clip actual: OJO_NEUTRO o ninguno (identidad),
        el que esté más cerca de la mediana del clip."""
        c = getattr(s, '_clip_obj', None)
        if c is None or 'LeftEye' not in c.idx:
            return np.eye(3)
        cache = getattr(s, '_neutros', {})
        s._neutros = cache
        if id(c) not in cache:
            r = {"Left": [], "Right": []}
            for t in np.linspace(0, c.dur, 24):
                Gc, _ = c.globales(t)
                for L in r:
                    r[L].append(_rv(Gc[c.idx['Head']].T @ Gc[c.idx[L + 'Eye']]))
            res = {}
            for L in r:
                med = np.median(np.array(r[L]), 0)
                usar = np.linalg.norm(med - np.array(OJO_NEUTRO[L])) < np.linalg.norm(med)
                res[L] = _rot(OJO_NEUTRO[L]) if usar else np.eye(3)
            cache[id(c)] = res
        return cache[id(c)][lado]

    def pose(s, clip, t, bucle=False, apply=True):
        s._t = float(t); s._clip = id(clip); s._clip_obj = clip
        G, Pm, b = s.globales(clip, t, bucle)
        w = s.pesos_cara(G, t, b)
        for k, v in s.cara_manual.items():
            w[k] = max(w.get(k, 0.0), float(v))
        s.base.set_morph(w)
        s.cara_actual = w
        return s.pose_mats(G, Pm, b, apply)

    def pesos_cara(s, G, t, b=None):
        p = s.pista or s._sin_pista
        w = p.pesos(t)
        ix = s.idx
        if s.OJOS and 'LeftEye' in ix:
            H = G[ix['Head']]; Ab = b['A'] if b is not None else None
            for lado, sg in (("Left", 1), ("Right", -1)):
                R = H.T @ G[ix[lado + 'Eye']] @ s._neutro_ojos(lado).T
                if Ab is not None:
                    A = Ab[ix['Head']]; R = A @ R @ A.T
                f = R @ np.array([0.0, 0.0, 1.0])
                yaw = np.degrees(np.arctan2(f[0], f[2])); pitch = np.degrees(np.arctan2(f[1], np.hypot(f[0], f[2])))
                if yaw >= 0:   # hacia la izquierda de Max (+X)
                    k = "eyeLookOut" + lado if sg > 0 else "eyeLookIn" + lado
                else:
                    k = "eyeLookIn" + lado if sg > 0 else "eyeLookOut" + lado
                w[k] = float(np.clip(w.get(k, 0) + abs(yaw) / OJOS_LADO, 0, 1))
                kp = ("eyeLookUp" if pitch >= 0 else "eyeLookDown") + lado
                w[kp] = float(np.clip(w.get(kp, 0) + abs(pitch) / OJOS_ARRIBA, 0, 1))
        # parpadeo extra en giros rápidos de cabeza
        fwd = G[ix['Head']] @ np.array([0.0, 0.0, 1.0])
        if s._ultimo is not None and 0 < t - s._ultimo[0] < 0.2:
            ang = np.degrees(np.arccos(np.clip(fwd @ s._ultimo[1], -1, 1))) / (t - s._ultimo[0])
            if ang > s.PARPADEO_EN_GIRO and (not len(p.parpadeos) or min(abs(p.parpadeos - t)) > 0.8):
                p.agregar_parpadeo(t)
        s._ultimo = (t, fwd)
        return w
