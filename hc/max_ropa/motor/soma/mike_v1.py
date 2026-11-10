"""Mike v1 (ZIP_35, archivo NUEVO: no cambia nada de lo que ya hay). PROTAGONISTA NUEVO: Mike (Mike Alger, CC-BY 4.0),
desde su archivo ORIGINAL (FBX de Sketchfab pasado a Personajes/Mike/mike_original.glb) con sus 50 expresiones.

Cómo funciona: Max sigue trabajando "por dentro" como director. Cada cuadro se calcula Max V12 completo (MaxCaraV12:
brazos afuera y planeados, pies clavados, parado derecho, balanceo, cabeza, dedos, manos en el objeto, cara viva...) y Mike
COPIA ese esqueleto ya corregido:
  - cada hueso de Mike toma la dirección del hueso de Max (con un segundo eje para el giro), como en las pruebas;
  - las MANOS van exactamente adonde están las manos de Max (IK de dos huesos): agarra los mismos objetos en el mismo
    sitio, y el codo apunta adonde apunta el de Max;
  - los PIES van adonde están los pies de Max (clavados), a la altura del tobillo de Mike;
  - la CARA: los pesos de la cara viva de Max (ARKit + visemas) se traducen a las 50 expresiones de Mike (tabla MAPA);
  - la MIRADA gira los huesos de los ojos de Mike (mismos grados que usa Max).
Lo que es de la ropa de Max (camiseta lisa, cintura, piel que no asoma) y la respiración NO se pasan (otra ropa).

Uso (igual que Max):
    import sys; sys.path.insert(0, '/home/claude')
    from soma import Clip
    from soma.mike_v1 import MikeCara, PistaCaraV12
    m = MikeCara(glb='/home/claude/Personajes/Mike/mike_original.glb', glb_max='/home/claude/Personajes/Max/max_v5.glb')
    m.pista = PistaCaraV12.desde_escena('<carpeta de salida de escena5.py>', '<escena.json>')
    res = m.pose(clip, t)        # lista (posiciones, normales) por pieza, como Max; m.prims tiene UV, textura, etc.
    m.joint_pos('Head')          # mismos nombres que Max (Head, Hips, Chest, LeftEye, LeftHand, LeftFoot, ...)
    m.cara_actual                # pesos de la cara de Max en este cuadro; m.cara_mike = los de Mike
"""
import os
import re
import numpy as np

from .max_v12 import MaxCaraV12, PistaCaraV12          # noqa: F401  (PistaCaraV12 se exporta para usarla igual)

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
MIKE_GLB = os.path.join(RAIZ, 'Personajes', 'Mike', 'mike_original.glb')
MAX_GLB = os.path.join(RAIZ, 'Personajes', 'Max', 'max_v5.glb')
Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
OJOS_LADO, OJOS_ARRIBA = 26.0, 20.0

# ------------------------------------------------------------------------------------------------ cara: Max -> Mike
MAPA = {}
def _m(k, *pares): MAPA.setdefault(k, []).extend(pares)
for _a in ('Left', 'Right'):
    _m('eyeBlink' + _a, ('Blink_' + _a, 1.0))
    _m('eyeSquint' + _a, ('Squint_' + _a, 0.35))
    _m('eyeWide' + _a, ('EyesWide_' + _a, 1.0))
    _m('browDown' + _a, ('BrowsDown_' + _a, 1.0), ('BrowsIn_' + _a, 0.4))
    _m('browOuterUp' + _a, ('BrowsUp_' + _a, 1.0))
    _m('cheekSquint' + _a, ('Squint_' + _a, 0.15))
    _m('mouthSmile' + _a, ('Smile_' + _a, 1.0))
    _m('mouthFrown' + _a, ('Frown_' + _a, 1.0))
    _m('mouthDimple' + _a, ('Smile_' + _a, 0.25))
    _m('mouthStretch' + _a, ('LowerLipDown_' + _a, 0.35), ('Frown_' + _a, 0.25))
    _m('mouthPress' + _a, ('UpperLipIn', 0.15), ('LowerLipIn', 0.15))
    _m('mouthLowerDown' + _a, ('LowerLipDown_' + _a, 1.0))
    _m('mouthUpperUp' + _a, ('UpperLipUp_' + _a, 1.0))
    _m('noseSneer' + _a, ('NoseScrunch_' + _a, 1.0))
_m('browInnerUp', ('BrowsUp_Left', 0.8), ('BrowsUp_Right', 0.8))
_m('cheekPuff', ('CheekPuff_Left', 1.0), ('CheekPuff_Right', 1.0))
_m('jawOpen', ('Jaw_Down', 1.0))
_m('jawForward', ('JawForeward', 1.0))
_m('jawLeft', ('Jaw_Left', 1.0)); _m('jawRight', ('Jaw_Right', 1.0))
_m('mouthClose', ('Jaw_Up', 0.3))
_m('mouthFunnel', ('MouthNarrow_Left', 0.6), ('MouthNarrow_Right', 0.6), ('MouthOpen', 0.2))
_m('mouthPucker', ('MouthWhistle_NarrowAdjust_Left', 0.8), ('MouthWhistle_NarrowAdjust_Right', 0.8),
   ('MouthNarrow_Left', 0.4), ('MouthNarrow_Right', 0.4))
_m('mouthLeft', ('Midmouth_Left', 1.0)); _m('mouthRight', ('Midmouth_Right', 1.0))
_m('mouthRollLower', ('LowerLipIn', 1.0)); _m('mouthRollUpper', ('UpperLipIn', 1.0))
_m('mouthShrugLower', ('LowerLipOut', 0.7)); _m('mouthShrugUpper', ('UpperLipOut', 0.7))
_m('tongueOut', ('TongueUp', 0.5))
_m('V_Open', ('Jaw_Down', 0.45), ('MouthOpen', 0.6))
_m('V_Wide', ('Smile_Left', 0.35), ('Smile_Right', 0.35), ('MouthOpen', 0.3), ('Jaw_Down', 0.15))
_m('V_Tight', ('MouthNarrow_Left', 0.6), ('MouthNarrow_Right', 0.6))
_m('V_Tight_O', ('MouthWhistle_NarrowAdjust_Left', 0.7), ('MouthWhistle_NarrowAdjust_Right', 0.7), ('MouthOpen', 0.15),
   ('Jaw_Down', 0.1))
_m('V_Explosive', ('UpperLipIn', 0.35), ('LowerLipIn', 0.35))
_m('V_Dental_Lip', ('LowerLipIn', 0.6), ('UpperLipUp_Left', 0.2), ('UpperLipUp_Right', 0.2))
_m('V_Affricate', ('MouthWhistle_NarrowAdjust_Left', 0.4), ('MouthWhistle_NarrowAdjust_Right', 0.4), ('MouthOpen', 0.25),
   ('Jaw_Down', 0.1))
_m('V_Lip_Open', ('MouthOpen', 0.5))


def _q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _marco(p, s):
    p = p / (np.linalg.norm(p) + 1e-12); s = s - p * (s @ p); s = s / (np.linalg.norm(s) + 1e-12)
    return np.stack([p, s, np.cross(p, s)], 1)


def _rot(eje, ang):
    eje = eje / (np.linalg.norm(eje) + 1e-12)
    K = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _dos_huesos(A, l1, l2, T, polo):
    """Codo/rodilla para que el extremo llegue a T (si alcanza), en el plano del 'polo'."""
    d_v = T - A; d = min(np.linalg.norm(d_v), (l1 + l2) * 0.9995); e = d_v / (np.linalg.norm(d_v) + 1e-12)
    x = (l1 * l1 - l2 * l2 + d * d) / (2 * d); h = np.sqrt(max(l1 * l1 - x * x, 0.0))
    pol = polo - e * (polo @ e)
    pol = pol / np.linalg.norm(pol) if np.linalg.norm(pol) > 1e-6 else np.cross(e, [1.0, 0, 0])
    return A + x * e + h * pol, A + d * e


class MikeCara:
    """Mike (protagonista) movido por Max V12 (esqueleto corregido + cara viva)."""
    MANOS_IK = True          # las manos van exactamente a las manos de Max (objetos agarrados en el mismo sitio)
    PIES_IK = True           # los pies van a los pies de Max (clavados)
    MIRADA = True            # los ojos giran según la mirada de Max
    CARA = True              # expresiones de Mike desde la cara de Max
    MANO_CUERPO = True       # la mano libre de Mike no se mete en SU cuerpo (chaqueta, camiseta, pantalón, cara, cuello)
    DEDOS_IK = True          # la mano que agarra: cada yema de Mike va a la yema de Max (toca el objeto igual que Max)
    DQ_BRAZOS = True         # piel y mangas de los brazos con cuaterniones duales (sin "caramelo" al girar el antebrazo)
    DQ_PIERNAS = True        # pantalón y piernas con cuaterniones duales (la entrepierna no se "aplasta" al abrir las piernas)
    ENTREPIERNA = True       # pesos nuevos en la entrepierna del pantalón (el original cambia de pierna en 1 cm y se rasga)
    ENTREPIERNA_CADERA = 0.35  # parte del peso de las piernas que pasa a la cadera arriba del tiro
    ENTREPIERNA_ANCHO = 0.03   # media transición izquierda/derecha (m)
    HOMBROS_SUAVES = True    # pesos suavizados en hombros, cuello y capucha de la chaqueta (sin "picos" al mover el brazo)
    HOMBROS_PASADAS = 25     # pasadas de suavizado
    PIEL_BAJO_MANGA = True   # la piel del brazo no asoma por la manga al doblar el codo
    CODO_HOLGURA = True      # la manga no entra en el cuerpo de Mike: gira el codo (la mano no se mueve), planeado
    CODO_GIRO_MAX = 40.0     # grados como máximo
    PLAN_VERSION = 5         # cambia si cambia la forma de planear (así no se usa un plan viejo guardado)
    PASO_PLANEADO = True     # caminar: la cadera baja SUAVE (planeada en toda la animación) y el pie de atrás levanta el
                             # talón (gira en la punta) cuando la pierna de Mike, más corta que la de Max, no llega
    TALON_MAX = 35.0         # grados que puede levantar el talón (girando sobre la punta del pie)
    PUNTA_MAX = 15.0         # grados que puede levantar la punta (pie de adelante, girando sobre el talón)
    BRAZO_SUAVE = 2.0        # cuadros (a 30 por segundo) de suavizado del brazo LIBRE: quita los tirones de 1-2 cuadros
                             # del brazo de Max (que en Mike doblan la tela del hombro); la mano que agarra no se toca
    CODO_TOLERANCIA = 0.005  # m que se permite que la manga apriete contra el cuerpo (tela contra tela)
    MANGA_DESDE = 0.5        # se revisa la manga desde la mitad del brazo hasta la muñeca (la axila no)
    DEDOS_MUNECA_MAX = 0.015  # m que se puede correr la muñeca que agarra para que todos los dedos toquen
    RADIOS = {'muneca': 0.028, 'palma': 0.016, 'dedo': 0.009, 'yema': 0.007}   # m de aire mínimo hasta la piel/ropa

    def __init__(s, glb=None, glb_max=None, **k_max):
        import gltf
        s.max = MaxCaraV12(glb=glb_max or MAX_GLB, **k_max)
        s.sk = gltf.Skinned(glb or MIKE_GLB)
        s.prims = s.sk.prims
        J = s.sk.J; s.N = N = J['nodes']; nn = len(N); s.par = par = s.sk.parent
        def lr(i):
            T_, R, S, M = s.sk.rest[i]
            if M is not None: return np.array(M, float).reshape(4, 4).T
            m = np.eye(4); m[:3, :3] = _q2m(R) @ np.diag(S); m[:3, 3] = T_; return m
        s.LR = [lr(i) for i in range(nn)]
        s.orden = []
        def vis(i):
            s.orden.append(i)
            for c in N[i].get('children', []): vis(c)
        for r in J['scenes'][J.get('scene', 0)]['nodes']: vis(r)
        s.WR = s._fk(lambda i, w: w)
        s.base_n = {}
        for i in s.sk.joints:
            s.base_n.setdefault(re.sub(r'_\d+$', '', N[i].get('name', '')).replace('mixamorig:', 'mixamorig_'), i)
        s._mapa_huesos()
        # cara
        s.p_cara = [p for p in s.prims if p['tnames']][0]
        s.TD = {nm: s.sk.acc(s.p_cara['targets'][k]['POSITION']) for k, nm in enumerate(s.p_cara['tnames'])
                if k < len(s.p_cara['targets'])}
        s.cara_mike = {}
        s.Wm = None
        s.info = {}
        if s.ENTREPIERNA: s._arreglar_entrepierna()
        if s.HOMBROS_SUAVES: s._suavizar_hombros()
        s._preparar_cuerpo()

    def _suavizar_hombros(s):
        """Chaqueta: en el modelo original la tela de arriba del hombro pasa de la espalda a la clavícula y al brazo en
        2-3 cm, y la capucha (que va encima del hombro) no sigue al hombro; al mover el brazo sale un "pico" en el hombro
        y la tela de abajo atraviesa la capucha. Aquí los pesos de esa zona se suavizan sobre la malla (cada punto toma
        un poco de sus vecinos, varias pasadas), con el borde de la zona fijo para que las mangas no cambien."""
        J = s.sk.J; nodos = [n.get('name', '') for n in J['nodes'] if 'mesh' in n]
        nom = [J['nodes'][j]['name'] for j in s.sk.joints]; nj = len(nom)
        T = s._T
        y0 = T('Spine2')[1]; xa = max(abs(T('LeftArm')[0] - T('Spine2')[0]), abs(T('RightArm')[0] - T('Spine2')[0]))
        Kr = np.stack([s.WR[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        s.info_hombros = 0
        for nodo in ('Sweatshirt', 'Shirt'):
            if nodo not in nodos: continue
            p = s.prims[nodos.index(nodo)]
            Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']]); V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P']) + Mv[:, :3, 3]
            nv = len(V)
            Wd = np.zeros((nv, nj)); np.add.at(Wd, (np.arange(nv)[:, None].repeat(p['J'].shape[1], 1), p['J']), p['W'])
            # vértices iguales (costuras de la textura) cuentan como uno
            _, uni, inv = np.unique(np.round(V, 5), axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
            nu = len(uni)
            F = inv[p['idx'].reshape(-1, 3)]
            a_ = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
            b_ = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
            from scipy.sparse import coo_matrix
            A = coo_matrix((np.ones(len(a_)), (a_, b_)), shape=(nu, nu)).tocsr(); A.data[:] = 1.0
            deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
            Vu = V[uni]; Wu = np.zeros((nu, nj)); np.add.at(Wu, inv, Wd); Wu /= np.bincount(inv, minlength=nu)[:, None]
            dx = np.abs(Vu[:, 0] - T('Spine2')[0])
            # zona: arriba del pecho, hasta un poco más afuera de la articulación del hombro; se suaviza más en el medio
            fuerza = (np.clip((Vu[:, 1] - (y0 - 0.02)) / 0.06, 0, 1) * np.clip(((xa + 0.07) - dx) / 0.06, 0, 1))
            zona = fuerza > 0
            if not zona.any(): continue
            for _ in range(s.HOMBROS_PASADAS):
                prom = (A @ Wu) / deg[:, None]
                Wu[zona] = Wu[zona] + 0.5 * fuerza[zona, None] * (prom[zona] - Wu[zona])
            Wn = Wu[inv]
            k = p['J'].shape[1]; orden = np.argsort(-Wn, 1)[:, :k]
            Wk = np.take_along_axis(Wn, orden, 1); Wk /= Wk.sum(1, keepdims=True)
            cambia = zona[inv]
            J2 = p['J'].copy(); W2 = p['W'].astype(np.float64).copy()
            J2[cambia] = orden[cambia]; W2[cambia] = Wk[cambia]
            p['J'] = J2.astype(p['J'].dtype); p['W'] = W2.astype(p['W'].dtype)
            s.info_hombros += int(cambia.sum())

    def _arreglar_entrepierna(s):
        """Pesos de la piel en la entrepierna (pantalón, piel, camisetas): en el modelo original la tela pasa de la pierna
        izquierda a la derecha en ~1 cm justo en la costura; al abrir o adelantar las piernas esos triángulos se estiran
        hasta 25 veces (la "tela clara" entre las piernas). Aquí el cambio de pierna es gradual (6 cm) y, arriba del
        tiro, parte del peso pasa a la cadera (como un pantalón de verdad)."""
        nom = [s.sk.J['nodes'][j]['name'] for j in s.sk.joints]
        def jx(n):
            c = [a for a, x in enumerate(nom) if x.endswith(n)]; return c[0] if c else None
        jh, jl, jr = jx('Hips'), jx('LeftUpLeg'), jx('RightUpLeg')
        if None in (jh, jl, jr): return
        T = s._T
        xl, xr = T('LeftUpLeg')[0], T('RightUpLeg')[0]; xm = (xl + xr) / 2; sgn = 1.0 if xl > xr else -1.0
        y_cad = (T('LeftUpLeg')[1] + T('RightUpLeg')[1]) / 2
        def sst(a, b, x):
            u = np.clip((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u)
        Kr = np.stack([s.WR[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        nj = len(nom); s.info_entrepierna = 0
        # altura del tiro: el punto más bajo del pantalón en la línea del medio
        for p in s.prims:
            Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']]); V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P']) + Mv[:, :3, 3]
            Wd = np.zeros((len(V), nj)); np.add.at(Wd, (np.arange(len(V))[:, None].repeat(p['J'].shape[1], 1), p['J']), p['W'])
            wl, wr = Wd[:, jl], Wd[:, jr]; piernas = wl + wr
            mezcla = (np.minimum(wl, wr) > 0.02)
            if not mezcla.any(): continue
            y_tiro = float(np.percentile(V[mezcla, 1], 2))
            x = (V[:, 0] - xm) * sgn
            zona = (piernas > 0.01) & (np.abs(x) < 0.08) & (V[:, 1] > y_tiro - 0.06) & (V[:, 1] < y_cad + 0.05)
            if not zona.any(): continue
            iz = np.where(zona)[0]
            # 1) izquierda/derecha gradual: 6 cm de transición centrada en la costura
            #    solo ARRIBA del tiro (abajo cada pierna es un tubo aparte y se queda con su pierna)
            arriba = sst(y_tiro, y_tiro + 0.025, V[iz, 1])
            fl0 = np.where(piernas[iz] > 1e-6, wl[iz] / np.maximum(piernas[iz], 1e-6), 0.5)
            fl = fl0 * (1 - arriba) + sst(-s.ENTREPIERNA_ANCHO, s.ENTREPIERNA_ANCHO, x[iz]) * arriba
            # 2) arriba del tiro y cerca del medio, hasta 55% del peso de las piernas pasa a la cadera
            fh = s.ENTREPIERNA_CADERA * (1 - sst(0.0, 0.07, np.abs(x[iz]))) * sst(y_tiro - 0.01, y_tiro + 0.07, V[iz, 1])
            tot = piernas[iz]
            Wd[iz, jh] += tot * fh
            Wd[iz, jl] = tot * (1 - fh) * fl; Wd[iz, jr] = tot * (1 - fh) * (1 - fl)
            k = p['J'].shape[1]
            orden = np.argsort(-Wd[iz], 1)[:, :k]
            Wn = np.take_along_axis(Wd[iz], orden, 1); Wn /= Wn.sum(1, keepdims=True)
            J2 = p['J'].copy(); W2 = p['W'].astype(np.float64).copy()
            J2[iz] = orden; W2[iz] = Wn
            p['J'] = J2.astype(p['J'].dtype); p['W'] = W2.astype(p['W'].dtype)
            s.info_entrepierna += len(iz)

    # -------------------------------------------------------------------------------------------- esqueleto
    def _fk(s, cambia):
        W = [None] * len(s.N)
        for i in s.orden:
            W[i] = cambia(i, (W[s.par[i]] if i in s.par else np.eye(4)) @ s.LR[i])
        return W

    def _i(s, n): return s.base_n['mixamorig_' + n]

    def _T(s, n): return s.WR[s._i(n)][:3, 3]

    def _mapa_huesos(s):
        T = s._T; H = {}
        def spec(tgt, sp, ss, tp, ts): H[s._i(tgt)] = (sp, ss, tp, ts)
        d = lambda a, b: (lambda P: P[b] - P[a])
        hom_s = d('RightArm', 'LeftArm'); hom_t = T('LeftArm') - T('RightArm')
        cad_s = d('RightLeg', 'LeftLeg'); cad_t = T('LeftUpLeg') - T('RightUpLeg')
        spec('Hips', d('Hips', 'Spine1'), cad_s, T('Spine') - T('Hips'), cad_t)
        spec('Spine', d('Spine1', 'Spine2'), cad_s, T('Spine1') - T('Spine'), cad_t)
        spec('Spine1', d('Spine2', 'Chest'), hom_s, T('Spine2') - T('Spine1'), hom_t)
        spec('Spine2', d('Chest', 'Neck1'), hom_s, T('Neck') - T('Spine2'), hom_t)
        spec('Neck', d('Neck1', 'Head'), hom_s, T('Head') - T('Neck'), hom_t)
        # cabeza: dirección de la T del esqueleto SOMA (HeadEnd-Head) y línea de los ojos
        from . import Clip
        tp = Clip(os.path.join(AQUI, 'somaskel77_standard_tpose.bvh')); _, PT = tp.globales(0.0)
        spec('Head', d('Head', 'HeadEnd'), d('RightEye', 'LeftEye'), PT[tp.idx['HeadEnd']] - PT[tp.idx['Head']],
             T('LeftEye') - T('RightEye'))
        for L in ('Left', 'Right'):
            palma_s = d(L + 'HandPinky2', L + 'HandIndex2'); palma_t = T(L + 'HandIndex1') - T(L + 'HandPinky1')
            spec(L + 'Shoulder', d(L + 'Shoulder', L + 'Arm'), d('Chest', 'Neck1'), T(L + 'Arm') - T(L + 'Shoulder'),
                 T('Neck') - T('Spine2'))
            spec(L + 'Arm', d(L + 'Arm', L + 'ForeArm'), d(L + 'ForeArm', L + 'Hand'), T(L + 'ForeArm') - T(L + 'Arm'),
                 T(L + 'Hand') - T(L + 'ForeArm'))
            spec(L + 'ForeArm', d(L + 'ForeArm', L + 'Hand'), palma_s, T(L + 'Hand') - T(L + 'ForeArm'), palma_t)
            spec(L + 'Hand', d(L + 'Hand', L + 'HandMiddle2'), palma_s, T(L + 'HandMiddle1') - T(L + 'Hand'), palma_t)
            for dd in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
                src = [f'{L}Hand{dd}{x}' for x in ((1, 2, 3, 'End') if dd == 'Thumb' else (2, 3, 4, 'End'))]
                tgt = [f'{L}Hand{dd}{x}' for x in (1, 2, 3)]
                if not all('mixamorig_' + t_ in s.base_n for t_ in tgt): continue
                for q in range(3):
                    tdir = T(tgt[q + 1]) - T(tgt[q]) if q < 2 else T(tgt[2]) - T(tgt[1])
                    spec(tgt[q], d(src[q], src[q + 1]), palma_s, tdir, palma_t)
            spec(L + 'UpLeg', d(L + 'Leg', L + 'Shin'), d(L + 'Shin', L + 'Foot'), T(L + 'Leg') - T(L + 'UpLeg'),
                 T(L + 'Foot') - T(L + 'Leg'))
            spec(L + 'Leg', d(L + 'Shin', L + 'Foot'), d(L + 'Foot', L + 'ToeBase'), T(L + 'Foot') - T(L + 'Leg'),
                 T(L + 'ToeBase') - T(L + 'Foot'))
            spec(L + 'Foot', d(L + 'Foot', L + 'ToeBase'), cad_s, T(L + 'ToeBase') - T(L + 'Foot'), cad_t)
        s.H = H; s.FR = {i: _marco(tp_, ts) for i, (_, _, tp_, ts) in H.items()}
        s.raiz = s._i('Hips')
        # medidas de Mike (metros, Y arriba)
        s.l_brazo = {L: (np.linalg.norm(T(L + 'ForeArm') - T(L + 'Arm')), np.linalg.norm(T(L + 'Hand') - T(L + 'ForeArm'))) for L in ('Left', 'Right')}
        s.l_pie = {L: float(np.linalg.norm(T(L + 'ToeBase') - T(L + 'Foot'))) for L in ('Left', 'Right')}
        s.l_pierna = {L: (np.linalg.norm(T(L + 'Leg') - T(L + 'UpLeg')), np.linalg.norm(T(L + 'Foot') - T(L + 'Leg'))) for L in ('Left', 'Right')}
        Kr = np.stack([s.WR[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        ys = []
        for p in s.prims:
            Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']]); ys.append(np.einsum('vj,vj->v', Mv[:, 1, :3], p['P']) + Mv[:, 1, 3])
        s.piso = float(np.percentile(np.concatenate(ys), 0.05))
        s.cadera_alto = float(T('Hips')[1] - s.piso)
        s.tobillo_alto = float((T('LeftFoot')[1] + T('RightFoot')[1]) / 2 - s.piso)
        # grosor del antebrazo CON la manga de la chaqueta (para que la manga no entre en el cuerpo)
        nom = [s.sk.J['nodes'][j]['name'] for j in s.sk.joints]
        nodos = [n.get('name', '') for n in s.sk.J['nodes'] if 'mesh' in n]
        s.r_antebrazo = {'Left': 0.045, 'Right': 0.045}; s.r_brazo = {'Left': 0.055, 'Right': 0.055}
        if 'Sweatshirt' in nodos:
            p = s.prims[nodos.index('Sweatshirt')]
            Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']]); Vr = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P']) + Mv[:, :3, 3]
            for L in ('Left', 'Right'):
                js = [a for a, n in enumerate(nom) if (L + 'ForeArm') in n]
                sel = (p['W'] * np.isin(p['J'], js)).sum(1) > 0.8
                if sel.sum() > 10:
                    E_, H_ = T(L + 'ForeArm'), T(L + 'Hand'); ax = (H_ - E_) / np.linalg.norm(H_ - E_)
                    d_ = Vr[sel] - E_; r_ = np.linalg.norm(d_ - np.outer(d_ @ ax, ax), axis=1)
                    s.r_antebrazo[L] = float(np.percentile(r_, 75))
                js = [a for a, n in enumerate(nom) if n.endswith(L + 'Arm')]
                sel = (p['W'] * np.isin(p['J'], js)).sum(1) > 0.8
                if sel.sum() > 10:
                    A_, E_ = T(L + 'Arm'), T(L + 'ForeArm'); ax = (E_ - A_) / np.linalg.norm(E_ - A_)
                    d_ = Vr[sel] - A_; r_ = np.linalg.norm(d_ - np.outer(d_ @ ax, ax), axis=1)
                    s.r_brazo[L] = float(np.percentile(r_, 75))

    # -------------------------------------------------------------------------------------------- Max -> fuente
    def _fuente(s):
        """Posiciones (Y arriba, metros, piso 0) de los 77 huesos SOMA de Max ya corregidos."""
        m = s.max
        return {n: Cz.T @ m.joint_pos(n) for n in m.idx}

    @property
    def pista(s): return s.max.pista

    @pista.setter
    def pista(s, v): s.max.pista = v

    @property
    def cara_actual(s): return s.max.cara_actual

    def set_morph(s, w): s.max.set_morph(w)

    def pesos_mike(s, w):
        out = {}
        for k, v in (w or {}).items():
            if not isinstance(v, (int, float, np.floating, np.integer)) or not v: continue
            for n, f in MAPA.get(k, ()):
                if n in s.TD: out[n] = out.get(n, 0.0) + float(v) * f
        return {n: float(np.clip(x, 0, 1)) for n, x in out.items()}

    def _max_alto(s):
        if not hasattr(s, '_mx_alto'):
            b = s.max.base; b.pose(None, 0, apply=True)
            cad = b.joint_pos('Hips'); pie = (b.joint_pos('LeftFoot') + b.joint_pos('RightFoot')) / 2
            s._mx_alto = (float(cad[2]), float(pie[2]))
        return s._mx_alto

    # -------------------------------------------------------------------------------------------- IK propio de Mike
    def _preparar_cuerpo(s):
        """Partes del cuerpo de Mike contra las que no puede entrar su mano: tronco de la chaqueta, camiseta, pantalón,
        cara y cuello (sin mangas ni manos)."""
        J = s.sk.J; nodos_malla = [n.get('name', '') for n in J['nodes'] if 'mesh' in n]
        nom = [J['nodes'][j]['name'] for j in s.sk.joints]
        def peso(p, claves):
            js = [a for a, n in enumerate(nom) if any(k in n for k in claves)]
            return (p['W'] * np.isin(p['J'], js)).sum(1)
        BRAZO = ('Shoulder', 'Arm', 'Hand')
        s.cuerpo = []
        for pi, (p, nodo) in enumerate(zip(s.prims, nodos_malla)):
            if nodo in ('Sweatshirt', 'Shirt'): sel = np.where(peso(p, BRAZO) < 0.2)[0]
            elif nodo == 'Pants': sel = np.arange(len(p['P']))
            elif nodo == 'Body': sel = np.where((peso(p, ('Head', 'Neck', 'Eye')) > 0.5) & (peso(p, BRAZO) < 0.1))[0]
            else: continue
            if len(sel): s.cuerpo.append((pi, sel))

    def _cuerpo(s, W):
        K = np.stack([W[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        Ps, Ns = [], []
        for pi, sel in s.cuerpo:
            p = s.prims[pi]
            Mv = np.einsum('vk,vkij->vij', p['W'][sel], K[p['J'][sel]])
            Ps.append(np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P'][sel]) + Mv[:, :3, 3])
            n = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N'][sel]); Ns.append(n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9))
        from scipy.spatial import cKDTree
        P = np.concatenate(Ps); return cKDTree(P), P, np.concatenate(Ns)

    def _sondas(s, W, L):
        """Puntos de la mano de Mike con su radio (muñeca, palma, nudillos, yemas)."""
        pos = lambda n: W[s._i(L + n)][:3, 3]
        pts, rad = [pos('Hand')], [s.RADIOS['muneca']]
        pts.append((pos('Hand') + pos('HandIndex1') + pos('HandPinky1')) / 3); rad.append(s.RADIOS['palma'])
        for dd in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
            for k_ in (2, 3, 4):
                n = f'Hand{dd}{k_}'
                if 'mixamorig_' + L + n in s.base_n:
                    pts.append(pos(n)); rad.append(s.RADIOS['yema'] if k_ == 4 else s.RADIOS['dedo'])
        return np.array(pts), np.array(rad)

    def _metida(s, W, L, cuerpo):
        arbol, P, N = cuerpo
        pts, rad = s._sondas(W, L)
        d, j = arbol.query(pts)
        sd = ((pts - P[j]) * N[j]).sum(1)
        pen = np.where(d < 0.08, rad - sd, -1.0)
        k = int(np.argmax(pen))
        return float(pen[k]), N[j[k]]

    PLAN_FPS = 30

    def _manga_puntos(s, W, L):
        """Vértices de la manga (brazo y antebrazo, sin la mano) de la chaqueta de Mike en esta pose."""
        if not hasattr(s, '_manga_sel'):
            nom = [s.sk.J['nodes'][j]['name'] for j in s.sk.joints]
            nodos = [n.get('name', '') for n in s.sk.J['nodes'] if 'mesh' in n]
            s._i_chaq = nodos.index('Sweatshirt') if 'Sweatshirt' in nodos else None
            s._manga_sel = {}
            if s._i_chaq is not None:
                p = s.prims[s._i_chaq]
                Kr = np.stack([s.WR[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
                Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']]); Vr = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P']) + Mv[:, :3, 3]
                for L_ in ('Left', 'Right'):
                    js = [a for a, n in enumerate(nom) if n.endswith(L_ + 'Arm') or n.endswith(L_ + 'ForeArm')]
                    A_, E_ = s._T(L_ + 'Arm'), s._T(L_ + 'ForeArm'); ax = (E_ - A_) / np.linalg.norm(E_ - A_)
                    lejos = ((Vr - A_) @ ax) > s.MANGA_DESDE * np.linalg.norm(E_ - A_)     # sin la axila (ahí la tela siempre toca)
                    s._manga_sel[L_] = np.where(((p['W'] * np.isin(p['J'], js)).sum(1) > 0.5) & lejos)[0]
        if s._i_chaq is None or not len(s._manga_sel.get(L, [])): return None
        p = s.prims[s._i_chaq]; sel = s._manga_sel[L]
        K = np.stack([W[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        Mv = np.einsum('vk,vkij->vij', p['W'][sel], K[p['J'][sel]])
        return np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P'][sel]) + Mv[:, :3, 3]

    def _giro_necesario(s, W, L, cuerpo):
        """Giro del codo (rad, con signo) para que la manga REAL no entre en el cuerpo de Mike (la mano queda fija)."""
        V = s._manga_puntos(W, L)
        if V is None: return 0.0
        arbol, Pc, Nc = cuerpo
        A = W[s._i(L + 'Arm')][:3, 3]; H = W[s._i(L + 'Hand')][:3, 3]; ax = (H - A) / np.linalg.norm(H - A)
        def pen(th):
            Vr = A + (V - A) @ _rot(ax, th).T
            d, j = arbol.query(Vr)
            sd = ((Vr - Pc[j]) * Nc[j]).sum(1)
            m_ = d < 0.03
            return float(-sd[m_].min()) if m_.any() else -1.0
        tol = s.CODO_TOLERANCIA
        p0 = pen(0.0)
        if p0 <= tol: return 0.0
        paso = np.radians(5); sg = 1.0 if pen(paso) < pen(-paso) else -1.0
        mejor, pm = 0.0, p0; ok = None
        for k_ in range(1, int(s.CODO_GIRO_MAX / 5) + 1):
            th = sg * k_ * paso; pv = pen(th)
            if pv < pm: mejor, pm = th, pv
            if pv <= tol: ok = th; break
        if ok is not None:
            lo, hi = ok - sg * paso, ok
            for _ in range(6):
                mid = (lo + hi) / 2
                if pen(mid) <= tol: hi = mid
                else: lo = mid
            mejor = hi
        return mejor

    def _plan_codo(s, clip, bucle=False):
        """Giro del codo de cada cuadro (30 por segundo) para toda la animación, suavizado (empieza antes y termina
        después, sin saltos). Se guarda en soma/planes_guardados/ como el plan de Max."""
        c = s.__dict__.setdefault('_planes_codo', {})
        if id(clip) in c: return c[id(clip)]
        import os, pickle, hashlib
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        h = hashlib.md5(); h.update(s.max._clave_plan(clip, 'mike_codo').encode())
        h.update(repr(sorted((k, repr(getattr(type(s), k))) for k in dir(type(s)) if k.isupper() and isinstance(getattr(type(s), k), (int, float, bool, dict)))).encode())
        ruta = os.path.join(s.max._carpeta_planes(), 'mike_codo_%s.pkl' % h.hexdigest()[:20])
        plan = None
        if s.max.GUARDAR_PLAN and os.path.exists(ruta):
            try: plan = pickle.load(open(ruta, 'rb'))
            except Exception: plan = None
        if plan is None:
            n = int(np.floor(clip.dur * s.PLAN_FPS)) + 1
            crudo = {'Left': np.zeros(n), 'Right': np.zeros(n)}
            for f in range(n):
                W, P, P2, AG, _ = s._esqueleto(clip, f / s.PLAN_FPS, bucle, cuadro=f)
                cuerpo = s._cuerpo(W)
                for L in crudo: crudo[L][f] = s._giro_necesario(W, L, cuerpo)
            plan = {}
            for L, th in crudo.items():
                env = maximum_filter1d(np.abs(th), size=15)         # empieza ~7 cuadros antes, termina ~7 después
                sg = np.sign(th); ult = 0.0
                for k_ in range(len(sg)):                               # el signo del giro más cercano
                    if sg[k_] != 0: ult = sg[k_]
                    else: sg[k_] = ult
                ult = 0.0
                for k_ in range(len(sg) - 1, -1, -1):
                    if th[k_] != 0: ult = np.sign(th[k_])
                    elif ult != 0 and env[k_] > 0: sg[k_] = ult
                plan[L] = gaussian_filter1d(env * sg, 5.0)
            if s.max.GUARDAR_PLAN:
                try: pickle.dump(plan, open(ruta, 'wb'))
                except Exception: pass
        c[id(clip)] = plan
        return plan

    def _pie_objetivo(s, P, L, tob_mx, ang):
        """Tobillo de Mike donde va el pie de Max; con ang > 0 el pie gira sobre la PUNTA (levanta el talón) y con
        ang < 0 gira sobre el TALÓN (levanta la punta). Devuelve (tobillo, giro del pie)."""
        obj = P[L + 'Foot'].copy(); obj[1] = obj[1] - tob_mx + s.tobillo_alto + s.piso
        if abs(ang) < 1e-6: return obj, np.eye(3)
        u = P[L + 'ToeBase'] - P[L + 'Foot']; uh = np.array([u[0], 0.0, u[2]])
        if np.linalg.norm(uh) < 1e-6: return obj, np.eye(3)
        uh /= np.linalg.norm(uh); eje = np.cross(np.array([0.0, 1.0, 0.0]), uh)    # eje a lo ancho del pie
        if ang > 0:
            piv = obj + u / (np.linalg.norm(u) + 1e-9) * s.l_pie[L]; R = _rot(eje, np.radians(ang))
        else:
            piv = obj - np.array([0.0, s.tobillo_alto * 0.85, 0.0]) - uh * 0.03; R = _rot(eje, np.radians(ang))
        return piv + R @ (obj - piv), R

    def _paso_crudo(s, P, W, tob_mx, angs=None):
        """Para un cuadro: giro del pie (talón/punta) y cuánto bajar la cadera para que la pierna de Mike llegue.
        Con angs (giros ya planeados) solo calcula cuánto bajar la cadera."""
        out = {}; baja = 0.0
        for L in ('Left', 'Right'):
            A = W[s._i(L + 'UpLeg')][:3, 3]; l1, l2 = s.l_pierna[L]; alc = (l1 + l2) * 0.993
            def dist(a): return np.linalg.norm(s._pie_objetivo(P, L, tob_mx, a)[0] - A)
            ang = 0.0
            if angs is not None: ang = angs[L]
            elif dist(0.0) > alc:
                # pie ATRÁS de la cadera: levanta el talón; pie ADELANTE: levanta la punta
                adel = np.cross(P['LeftLeg'] - P['RightLeg'], np.array([0.0, 1.0, 0.0]))
                atras = float((P[L + 'Foot'] - (P['LeftLeg'] + P['RightLeg']) / 2) @ adel) < 0
                cands = ([float(x) for x in np.arange(2.5, s.TALON_MAX + 0.1, 2.5)] if atras else
                         [-float(x) for x in np.arange(2.5, s.PUNTA_MAX + 0.1, 2.5)])
                ds = [dist(a) for a in cands]
                ok = [a for a, d_ in zip(cands, ds) if d_ <= alc]
                ang = ok[0] if ok else (cands[int(np.argmin(ds))] if min(ds) < dist(0.0) else 0.0)
            v = s._pie_objetivo(P, L, tob_mx, ang)[0] - A; bl = 0.0
            if np.linalg.norm(v) > alc:
                hz = np.hypot(v[0], v[2]); bl = abs(v[1]) - np.sqrt(max(alc ** 2 - hz ** 2, 0.0))
            baja = max(baja, bl)
            out[L] = ang; out['baja_' + L] = bl; out['pie_' + L] = P[L + 'Foot'].copy()
        out['baja'] = baja
        return out

    def _plan_paso(s, clip, bucle=False):
        """Plan de los pies y la cadera en toda la animación (30 por segundo), suavizado: la cadera empieza a bajar
        antes y sube después (sin saltos), el talón sube y baja suave. Se guarda en soma/planes_guardados/."""
        c = s.__dict__.setdefault('_planes_paso', {})
        if id(clip) in c: return c[id(clip)]
        import os, pickle, hashlib
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        h = hashlib.md5(); h.update(s.max._clave_plan(clip, 'mike_paso').encode())
        h.update(repr((s.TALON_MAX, s.PUNTA_MAX, s.PLAN_FPS, s.MANOS_IK, s.l_pierna, s.tobillo_alto, s.BRAZO_SUAVE, s.PLAN_VERSION)).encode())
        ruta = os.path.join(s.max._carpeta_planes(), 'mike_paso_%s.pkl' % h.hexdigest()[:20])
        plan = None
        if s.max.GUARDAR_PLAN and os.path.exists(ruta):
            try: plan = pickle.load(open(ruta, 'rb'))
            except Exception: plan = None
        if plan is None:
            n = int(np.floor(clip.dur * s.PLAN_FPS)) + 1
            cr = {'Left': np.zeros(n), 'Right': np.zeros(n), 'baja': np.zeros(n)}
            rel, ag, crs = [], [], []
            for f in range(n):
                o = s._esqueleto(clip, f / s.PLAN_FPS, bucle, crudo=True, cuadro=f)
                for k_ in cr: cr[k_][f] = o[k_]
                rel.append(o['rel']); ag.append(o['agarra']); crs.append(o)
            tob0 = min(min(o_['pie_Left'][1], o_['pie_Right'][1]) for o_ in crs)
            # pie APOYADO (Max lo tiene quieto y abajo): solo ahí gira el pie sobre la punta o el talón, y solo ahí la
            # pierna de Mike TIENE que llegar (en el aire, si le faltan unos cm no se nota y la cadera no baja por eso)
            apoyo = {}
            for L in ('Left', 'Right'):
                pp = np.array([o_['pie_' + L] for o_ in crs])
                vel = np.r_[0.0, np.linalg.norm(np.diff(pp[:, [0, 2]], axis=0), axis=1)] * s.PLAN_FPS
                vel = np.minimum(vel, np.r_[vel[1:], vel[-1]])
                quieto = ((vel < 0.35) & (pp[:, 1] < tob0 + 0.10)).astype(float)
                apoyo[L] = np.clip(gaussian_filter1d(quieto, 1.5), 0, 1)
            plan = {L: gaussian_filter1d(cr[L] * apoyo[L], 1.5) * apoyo[L] for L in ('Left', 'Right')}
            plan['crudo'] = cr; plan['apoyo'] = apoyo
            # con el pie ya suavizado, cuánto tiene que bajar la cadera en cada cuadro (segunda pasada, rápida: Max
            # ya está guardado) y una curva suave que nunca quede por debajo de lo necesario
            nec = np.zeros(n)
            for f in range(n):
                o_ = s._esqueleto(clip, f / s.PLAN_FPS, bucle, crudo=True, cuadro=f,
                                  angs={'Left': plan['Left'][f], 'Right': plan['Right'][f]})
                nec[f] = max(o_['baja_Left'] * min(1.0, 0.25 + apoyo['Left'][f]), o_['baja_Right'] * min(1.0, 0.25 + apoyo['Right'][f]))
            baja = gaussian_filter1d(maximum_filter1d(nec, size=9), 3.0)
            for _ in range(6):
                falta = nec - baja
                if falta.max() < 0.0005: break
                baja = baja + 1.15 * gaussian_filter1d(maximum_filter1d(np.maximum(falta, 0), size=9), 3.0)
            plan['baja'] = baja; cr['necesita'] = nec
            # brazo libre: diferencia entre el brazo suavizado y el de Max (se suma en cada cuadro)
            if s.BRAZO_SUAVE > 0:
                nombres = sorted(rel[0]); R_ = np.array([[r[k_] for k_ in nombres] for r in rel])     # (n, j, 3)
                D = gaussian_filter1d(R_, s.BRAZO_SUAVE, axis=0, mode='nearest') - R_
                for L in ('Left', 'Right'):
                    a_ = gaussian_filter1d(maximum_filter1d(np.array([g[L] for g in ag]), size=7), 2.0)
                    sel = [i for i, k_ in enumerate(nombres) if k_.startswith(L)]
                    D[:, sel] *= np.clip(1.0 - a_, 0, 1)[:, None, None]
                plan['brazo'] = (nombres, D)
            if s.max.GUARDAR_PLAN:
                try: pickle.dump(plan, open(ruta, 'wb'))
                except Exception: pass
        c[id(clip)] = plan
        return plan

    def _dedos_ik(s, W, P, L, peso):
        """CCD en cada dedo: la yema de Mike va a la yema de Max (que ya toca el objeto)."""
        err0, err1 = [], []
        for dd in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
            cad = [f'{L}Hand{dd}{k_}' for k_ in (1, 2, 3, 4)]
            fin = f'{L}Hand{dd}End'
            if not all('mixamorig_' + c in s.base_n for c in cad) or fin not in P: continue
            ids = [s._i(c) for c in cad]; T = P[fin] + s._piso_v()
            err0.append(np.linalg.norm(W[ids[-1]][:3, 3] - T))
            for _ in range(6):
                for q in (2, 1, 0):
                    pj = W[ids[q]][:3, 3]; a = W[ids[-1]][:3, 3] - pj; b = T - pj
                    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6: continue
                    eje = np.cross(a, b); sn = np.linalg.norm(eje)
                    if sn < 1e-9: continue
                    ang = np.arctan2(sn, a @ b) * peso
                    R = _rot(eje / sn, min(ang, np.radians(20)))
                    for jj in ids[q:]:
                        W[jj] = W[jj].copy(); W[jj][:3, :3] = R @ W[jj][:3, :3]; W[jj][:3, 3] = pj + R @ (W[jj][:3, 3] - pj)
            err1.append(np.linalg.norm(W[ids[-1]][:3, 3] - T))
        return (max(err0) if err0 else 0.0), (max(err1) if err1 else 0.0)

    def _esqueleto(s, clip, t, bucle=False, crudo=False, cuadro=None, angs=None):
        """Esqueleto de Mike copiado de Max + IK de manos, pies y mano libre fuera del cuerpo."""
        plan_p = None
        if not crudo: s.info = {}
        if s.PIES_IK and s.PASO_PLANEADO and not crudo:
            plan_p = s._plan_paso(clip, bucle)          # antes de mover a Max (el plan lo mueve a otros tiempos)
        cm = s.__dict__.setdefault('_cache_max', {})
        clave = (id(clip), cuadro, bucle)
        if cuadro is not None and clave in cm:
            P, libres = cm[clave]; s.max._dedos_libres = dict(libres)
        else:
            s.max.pose(clip, t, bucle, apply=True)
            P = s._fuente()
            if cuadro is not None: cm[clave] = (P, dict(getattr(s.max, '_dedos_libres', {}) or {}))
        if plan_p is not None and 'brazo' in plan_p:
            nombres, D = plan_p['brazo']; fq = float(t) * s.PLAN_FPS
            i0 = int(np.clip(np.floor(fq), 0, len(D) - 1)); i1 = min(i0 + 1, len(D) - 1); w1 = float(np.clip(fq - i0, 0, 1))
            d_ = D[i0] * (1 - w1) + D[i1] * w1
            P = dict(P)
            for k_, n in enumerate(nombres):
                if n in P: P[n] = P[n] + d_[k_]
        cad_mx, tob_mx = s._max_alto()
        kH = s.cadera_alto / cad_mx
        raiz_p = P['Hips'].copy(); raiz_p[1] = raiz_p[1] * kH + s.piso
        def mundo(P):
            def cambia(i, w):
                if i in s.H:
                    sp, ss, _, _ = s.H[i]
                    w = w.copy(); w[:3, :3] = _marco(sp(P), ss(P)) @ s.FR[i].T @ s.WR[i][:3, :3]
                if i == s.raiz:
                    w = w.copy(); w[:3, 3] = raiz_p
                return w
            return s._fk(cambia)
        W = mundo(P)
        ang_pie = {'Left': 0.0, 'Right': 0.0}
        if crudo:
            o = s._paso_crudo(P, W, tob_mx, angs) if s.PIES_IK else {'Left': 0.0, 'Right': 0.0, 'baja': 0.0}
            if angs is not None: return o
            o['rel'] = {n: P[n] - P[n[:4] + 'Arm' if n.startswith('Left') else 'RightArm'] for n in P
                        if (n.startswith('LeftHand') or n.startswith('RightHand') or n in ('LeftForeArm', 'RightForeArm'))}
            o['agarra'] = {L: float(1.0 - np.clip((getattr(s.max, '_dedos_libres', {}).get(L, 1.0) - 0.48) / 0.3, 0, 1))
                           for L in ('Left', 'Right')}
            return o
        if s.PIES_IK and plan_p is not None:
            # caminar: giro del pie y bajada de la cadera PLANEADOS (suaves) en toda la animación
            f = float(t) * s.PLAN_FPS; xs = np.arange(len(plan_p['baja']))
            ang_pie = {L: float(np.interp(f, xs, plan_p[L])) for L in ('Left', 'Right')}
            baja = float(np.interp(f, xs, plan_p['baja']))
            if baja > 1e-4:
                raiz_p[1] -= baja; W = mundo(P)
            s.info['cadera_baja_mm'] = round(1000 * baja, 1)
            s.info['talon_grados'] = {L: round(a, 1) for L, a in ang_pie.items()}
        elif s.PIES_IK:
            # si la pierna de Mike no llega al pie de Max, se baja un poco la cadera (si no, el pie se arrastra)
            baja = 0.0
            for L in ('Left', 'Right'):
                A = W[s._i(L + 'UpLeg')][:3, 3]; l1, l2 = s.l_pierna[L]
                obj = P[L + 'Foot'].copy(); obj[1] = obj[1] - tob_mx + s.tobillo_alto + s.piso
                v = obj - A; alcance = (l1 + l2) * 0.995
                if np.linalg.norm(v) > alcance:
                    hz = np.hypot(v[0], v[2]); necesita = abs(v[1]) - np.sqrt(max(alcance ** 2 - hz ** 2, 0.0))
                    baja = max(baja, necesita)
            if baja > 1e-4:
                raiz_p[1] -= baja; W = mundo(P)
        # IK: manos a las manos de Max, pies a los pies de Max (con el codo/rodilla apuntando como los de Max)
        AGARRA = {}
        for L in ('Left', 'Right'):
            x_ = float(np.clip((getattr(s.max, '_dedos_libres', {}).get(L, 1.0) - 0.48) / 0.3, 0, 1))
            AGARRA[L] = 1.0 - x_ * x_ * (3 - 2 * x_)
        P2 = dict(P)
        if s.MANOS_IK or s.PIES_IK:
            for L in ('Left', 'Right'):
                agarra = AGARRA[L]
                if s.MANOS_IK and agarra > 1e-3:
                    # solo la mano que AGARRA va al sitio exacto de la mano de Max (el objeto); la libre copia direcciones
                    A = W[s._i(L + 'Arm')][:3, 3]; l1, l2 = s.l_brazo[L]
                    propia = W[s._i(L + 'Hand')][:3, 3]
                    obj = propia + (P[L + 'Hand'] + s._piso_v() - propia) * agarra
                    polo = P[L + 'ForeArm'] - (P[L + 'Arm'] + P[L + 'Hand']) / 2
                    codo, mano = _dos_huesos(A, l1, l2, obj, polo)
                    # solo cuentan las direcciones: se arma el brazo desde el hombro de Max y la mano se lleva sus dedos
                    dm = (P[L + 'Arm'] + (mano - A)) - P[L + 'Hand']
                    P2[L + 'ForeArm'] = P[L + 'Arm'] + (codo - A)
                    for n in P:
                        if n.startswith(L + 'Hand'): P2[n] = P[n] + dm
                if s.PIES_IK:
                    A = W[s._i(L + 'UpLeg')][:3, 3]; l1, l2 = s.l_pierna[L]
                    obj, Rp = s._pie_objetivo(P, L, tob_mx, ang_pie[L])
                    polo = P[L + 'Shin'] - (P[L + 'Leg'] + P[L + 'Foot']) / 2
                    rod, pie = _dos_huesos(A, l1, l2, obj, polo)
                    if not crudo: s.info['pie_falta_mm_' + L] = round(1000 * float(np.linalg.norm(pie - obj)), 1)
                    P2[L + 'Shin'] = P[L + 'Leg'] + (rod - A)
                    P2[L + 'Foot'] = P[L + 'Leg'] + (pie - A)
                    for n in (L + 'ToeBase', L + 'ToeEnd'):          # el pie gira (talón o punta) si hace falta
                        if n in P: P2[n] = P2[L + 'Foot'] + Rp @ (P[n] - P[L + 'Foot'])
            W = mundo(P2)
        # mano LIBRE: no se mete en el cuerpo de Mike (su chaqueta es más gruesa que la camiseta de Max)
        if s.MANO_CUERPO and any(AGARRA[L] < 0.5 for L in AGARRA):
            cuerpo = s._cuerpo(W)
            for L in ('Left', 'Right'):
                if AGARRA[L] >= 0.5: continue
                pen0, _ = s._metida(W, L, cuerpo); pen = pen0
                for _ in range(4):
                    pen, n = s._metida(W, L, cuerpo)
                    if pen <= 0.0005: break
                    A = W[s._i(L + 'Arm')][:3, 3]; l1, l2 = s.l_brazo[L]
                    mano0 = W[s._i(L + 'Hand')][:3, 3]; codo0 = W[s._i(L + 'ForeArm')][:3, 3]
                    obj = mano0 + n * (pen + 0.002)
                    codo, mano = _dos_huesos(A, l1, l2, obj, codo0 - (A + mano0) / 2)
                    base_a = P2[L + 'Arm']
                    dm = (base_a + (mano - A)) - P2[L + 'Hand']
                    P2[L + 'ForeArm'] = base_a + (codo - A)
                    for nn_ in list(P2):
                        if nn_.startswith(L + 'Hand'): P2[nn_] = P2[nn_] + dm
                    W = mundo(P2)
                s.info['mano_cuerpo_mm_' + L] = (round(1000 * max(pen0, 0), 1), round(1000 * max(pen, 0), 1))
        return W, P, P2, AGARRA, mundo

    def pose(s, clip, t, bucle=False, apply=True):
        if s.PIES_IK and s.PASO_PLANEADO:
            s._plan_paso(clip, bucle)                 # pies y cadera al caminar (se calcula o se lee del disco)
        if s.CODO_HOLGURA:
            plan = s._plan_codo(clip, bucle)          # se calcula (o se lee del disco) ANTES de este cuadro
        s.__dict__.pop('_cache_max', None)
        W, P, P2, AGARRA, mundo = s._esqueleto(clip, t, bucle)
        # manga contra el cuerpo de Mike: el codo gira (la mano no se mueve), PLANEADO sobre toda la animación (sin saltos)
        if s.CODO_HOLGURA:
            f = float(t) * s.PLAN_FPS
            for L in ('Left', 'Right'):
                th = float(np.interp(f, np.arange(len(plan[L])), plan[L]))
                s.info['codo_giro_' + L] = round(float(np.degrees(abs(th))), 1)
                if abs(th) > np.radians(0.3):
                    A = W[s._i(L + 'Arm')][:3, 3]; E = W[s._i(L + 'ForeArm')][:3, 3]; H = W[s._i(L + 'Hand')][:3, 3]
                    ax = (H - A) / np.linalg.norm(H - A)
                    P2[L + 'ForeArm'] = P2[L + 'Arm'] + _rot(ax, th) @ (E - A)
                    W = mundo(P2)
        # mano que AGARRA: las yemas de Mike van a las yemas de Max
        if s.DEDOS_IK:
            for L in ('Left', 'Right'):
                if AGARRA[L] > 0.05:
                    Wp = [w for w in W]
                    e0, e1 = s._dedos_ik(Wp, P, L, AGARRA[L])
                    # si algún dedo de Mike no llega a la yema de Max (dedos de otro largo), la muñeca se corre un poco
                    # (hasta DEDOS_MUNECA_MAX) hacia donde faltan, y se vuelve a acomodar cada dedo
                    falta = []
                    for dd in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
                        fin = f'{L}Hand{dd}End'; c4 = f'{L}Hand{dd}4'
                        if fin in P and 'mixamorig_' + c4 in s.base_n:
                            v = P[fin] + s._piso_v() - Wp[s._i(c4)][:3, 3]
                            if np.linalg.norm(v) > 0.002: falta.append(v)
                    if falta and e1 > 0.002:
                        corr = np.mean(falta, 0) * AGARRA[L]
                        if np.linalg.norm(corr) > s.DEDOS_MUNECA_MAX: corr *= s.DEDOS_MUNECA_MAX / np.linalg.norm(corr)
                        A = W[s._i(L + 'Arm')][:3, 3]; l1, l2 = s.l_brazo[L]
                        mano0 = W[s._i(L + 'Hand')][:3, 3]; codo0 = W[s._i(L + 'ForeArm')][:3, 3]
                        codo, mano = _dos_huesos(A, l1, l2, mano0 + corr, codo0 - (A + mano0) / 2)
                        base_a = P2[L + 'Arm']; dm = (base_a + (mano - A)) - P2[L + 'Hand']
                        P2[L + 'ForeArm'] = base_a + (codo - A)
                        for nn_ in list(P2):
                            if nn_.startswith(L + 'Hand'): P2[nn_] = P2[nn_] + dm
                        W = mundo(P2)
                        Wp = [w for w in W]
                        _, e1 = s._dedos_ik(Wp, P, L, AGARRA[L])
                    W = Wp
                    s.info['dedos_mm_' + L] = (round(1000 * e0, 1), round(1000 * e1, 1))
        if s.MIRADA:
            s._ojos(W, s.max.cara_actual or {})
        s.Wm = W
        # cara
        wm = s.pesos_mike(s.max.cara_actual) if s.CARA else {}
        s.cara_mike = wm
        K = np.stack([W[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        usa_dq = s.DQ_BRAZOS or s.DQ_PIERNAS
        if usa_dq:
            from .max_v9 import _dq_de_matrices, _matrices_de_dq
            KQ, KD = _dq_de_matrices(K)
            clave = (s.DQ_BRAZOS, s.DQ_PIERNAS)
            if getattr(s, '_dq_clave', None) != clave:
                nom = [s.sk.J['nodes'][j]['name'] for j in s.sk.joints]
                # brazos: solo antebrazo y mano (ahí es donde el giro "aplasta" la manga); el hombro con cuaterniones duales
                # hacía una "aleta" de tela al mover el brazo hacia adelante, así que el hombro y la espalda van normales
                claves = (('ForeArm', 'Hand') if s.DQ_BRAZOS else ()) + (('UpLeg', 'Leg', 'Foot', 'Toe') if s.DQ_PIERNAS else ())
                js = [a for a, n in enumerate(nom) if any(k in n for k in claves)]
                jt = [a for a, n in enumerate(nom) if any(k in n for k in ('Spine', 'Shoulder', 'Neck'))]
                s._dq_mask = [((p['W'] * np.isin(p['J'], js)).sum(1) > 0.01) & ((p['W'] * np.isin(p['J'], jt)).sum(1) < 0.01)
                              for p in s.prims]; s._dq_clave = clave
        Vs, Ns = [], []
        for k_p, p in enumerate(s.prims):
            Pp = p['P']
            if p is s.p_cara and wm:
                Pp = Pp.astype(np.float64).copy()
                for n, x in wm.items(): Pp += x * s.TD[n]
            Mv = np.einsum('vk,vkij->vij', p['W'], K[p['J']])
            R_, t_ = Mv[:, :3, :3], Mv[:, :3, 3]
            if usa_dq and s._dq_mask[k_p].any():
                # brazos con cuaterniones duales: la manga y la piel no se "aplastan" al girar el antebrazo
                sel = s._dq_mask[k_p]; Jm, Wm_ = p['J'][sel], p['W'][sel]
                q = KQ[Jm]; d = KD[Jm]
                sg = np.sign(np.einsum('vkc,vc->vk', q, q[np.arange(len(Jm)), np.argmax(Wm_, 1)])); sg[sg == 0] = 1
                Rd, td = _matrices_de_dq(np.einsum('vk,vkc->vc', Wm_ * sg, q), np.einsum('vk,vkc->vc', Wm_ * sg, d))
                R_ = R_.copy(); t_ = t_.copy(); R_[sel] = Rd; t_[sel] = td
            V = np.einsum('vij,vj->vi', R_, Pp) + t_
            Nn = np.einsum('vij,vj->vi', R_, p['N'])
            Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            Vs.append(V); Ns.append(Nn)
        if s.PIEL_BAJO_MANGA:
            s._piel_bajo_manga(Vs, Ns)
        res = []
        for p, V, Nn in zip(s.prims, Vs, Ns):
            V = V @ Cz.T - np.array([0, 0, s.piso]); Nn = Nn @ Cz.T
            res.append((V[p['idx']], Nn[p['idx']]))
        return res

    def _preparar_mangas(s):
        """Piel del brazo de Mike que en reposo queda tapada por la manga de la chaqueta (o de la camiseta)."""
        from scipy.spatial import cKDTree
        J = s.sk.J; nodos = [n.get('name', '') for n in J['nodes'] if 'mesh' in n]
        nom = [J['nodes'][j]['name'] for j in s.sk.joints]
        Kr = np.stack([s.WR[jn] @ s.sk.ibm[a] for a, jn in enumerate(s.sk.joints)])
        def rep(pi):
            p = s.prims[pi]; Mv = np.einsum('vk,vkij->vij', p['W'], Kr[p['J']])
            V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['P']) + Mv[:, :3, 3]
            n = np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N']); return V, n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        s.i_piel = nodos.index('Body'); s.i_telas = [nodos.index(n) for n in ('Sweatshirt', 'Shirt') if n in nodos]
        p = s.prims[s.i_piel]
        js = [a for a, n in enumerate(nom) if any(k in n for k in ('Arm', 'Shoulder'))]
        brazo = (p['W'] * np.isin(p['J'], js)).sum(1) > 0.3
        Vp, _ = rep(s.i_piel)
        Vt = np.concatenate([rep(i)[0] for i in s.i_telas]); Nt = np.concatenate([rep(i)[1] for i in s.i_telas])
        d, j = cKDTree(Vt).query(Vp)
        dentro = ((Vp - Vt[j]) * Nt[j]).sum(1) < 0.002
        # además: toda la piel del brazo que está entre el hombro y el puño de la manga (1 cm antes del borde)
        ich = nodos.index('Sweatshirt'); pc = s.prims[ich]; Vc, _ = rep(ich)
        hasta = np.zeros(len(Vp), bool)
        for L in ('Left', 'Right'):
            codo = s.WR[s._i(L + 'ForeArm')][:3, 3]; mu = s.WR[s._i(L + 'Hand')][:3, 3]; ax = (mu - codo) / np.linalg.norm(mu - codo)
            jl = [a for a, n in enumerate(nom) if (L + 'ForeArm') in n]
            manga = (pc['W'] * np.isin(pc['J'], jl)).sum(1) > 0.3
            if not manga.any(): continue
            borde = ((Vc[manga] - codo) @ ax).max()
            jsl = [a for a, n in enumerate(nom) if any((L + k) in n for k in ('Arm', 'ForeArm', 'Shoulder'))]
            del_lado = (p['W'] * np.isin(p['J'], jsl)).sum(1) > 0.3
            hasta |= del_lado & (((Vp - codo) @ ax) < borde - 0.01)
        s.tapada = np.where(brazo & ((dentro & (d < 0.04)) | hasta))[0]

    def _piel_bajo_manga(s, Vs, Ns):
        """La piel tapada por la manga se queda adentro (4 mm bajo la tela) aunque el codo doble."""
        from scipy.spatial import cKDTree
        if not hasattr(s, 'tapada'): s._preparar_mangas()
        if not len(s.tapada): return
        Vt = np.concatenate([Vs[i] for i in s.i_telas]); Nt = np.concatenate([Ns[i] for i in s.i_telas])
        Vp = Vs[s.i_piel]; q = Vp[s.tapada]
        d, j = cKDTree(Vt).query(q)
        sd = ((q - Vt[j]) * Nt[j]).sum(1)
        mal = (sd > -0.004) & (d < 0.05)
        if mal.any():
            Vp = Vp.copy(); Vp[s.tapada[mal]] -= Nt[j[mal]] * (sd[mal] + 0.004)[:, None]; Vs[s.i_piel] = Vp
        s.info['piel_fuera_de_manga'] = int(mal.sum())

    def _piso_v(s): return np.array([0.0, s.piso, 0.0])

    def _ojos(s, W, w):
        up = W[s._i('Head')][:3, 3] - W[s._i('Neck')][:3, 3]; up /= np.linalg.norm(up)
        izq = W[s._i('LeftArm')][:3, 3] - W[s._i('RightArm')][:3, 3]; izq -= up * (izq @ up); izq /= np.linalg.norm(izq)
        for lado, sg in (('Left', 1), ('Right', -1)):
            if 'mixamorig_' + lado + 'Eye' not in s.base_n: continue
            a_izq = w.get('eyeLookOut' + lado, 0) if sg > 0 else w.get('eyeLookIn' + lado, 0)
            a_der = w.get('eyeLookIn' + lado, 0) if sg > 0 else w.get('eyeLookOut' + lado, 0)
            yaw = np.radians((a_izq - a_der) * OJOS_LADO)
            pitch = np.radians((w.get('eyeLookUp' + lado, 0) - w.get('eyeLookDown' + lado, 0)) * OJOS_ARRIBA)
            R = _rot(up, yaw) @ _rot(izq, -pitch)
            j = s._i(lado + 'Eye'); W[j] = W[j].copy(); W[j][:3, :3] = R @ W[j][:3, :3]

    # -------------------------------------------------------------------------------------------- posiciones
    SOMA_A_MIKE = {'Hips': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1', 'Chest': 'Spine2', 'Neck1': 'Neck', 'Neck2': 'Neck',
                   'Head': 'Head', 'LeftEye': 'LeftEye', 'RightEye': 'RightEye'}
    for _L in ('Left', 'Right'):
        SOMA_A_MIKE.update({_L + 'Shoulder': _L + 'Shoulder', _L + 'Arm': _L + 'Arm', _L + 'ForeArm': _L + 'ForeArm',
                            _L + 'Hand': _L + 'Hand', _L + 'Leg': _L + 'UpLeg', _L + 'Shin': _L + 'Leg',
                            _L + 'Foot': _L + 'Foot', _L + 'ToeBase': _L + 'ToeBase'})

    def joint_pos(s, nombre):
        """Posición (Z arriba, metros, piso 0) de un hueso con los nombres de Max/SOMA."""
        mk = s.SOMA_A_MIKE.get(nombre)
        if s.Wm is not None and mk and ('mixamorig_' + mk) in s.base_n:
            return Cz @ (s.Wm[s._i(mk)][:3, 3] - s._piso_v())
        return s.max.joint_pos(nombre)
