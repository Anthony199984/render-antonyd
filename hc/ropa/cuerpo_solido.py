"""CUERPO SÓLIDO: Max no se atraviesa a sí mismo (ni a cajas/piso). Es la ÚLTIMA capa de pose del motor.

Cómo funciona (todo dentro del motor, sin archivos intermedios):
 1. Se toma la pose FINAL del motor (después de todas sus capas) en cada cuadro del clip.
 2. Se arma un muñeco de física (MuJoCo) con la FORMA REAL de Max: cada hueso lleva la envolvente de la piel que mueve
    (no cápsulas a ojo) y está exactamente donde el motor pone ese hueso.
 3. Se buscan los momentos en que un brazo se mete en el cuerpo (o algo se mete en una caja / el piso).
 4. SOLO en esos momentos (con medio segundo de entrada y salida suaves) el brazo se vuelve "de física": choca con el
    cuerpo en vez de atravesarlo y sigue la animación lo más cerca que puede. El resto del cuerpo y el resto del
    tiempo quedan EXACTAMENTE como los dejó el motor (agarres incluidos).

Uso:
    from cuerpo_solido import con_cuerpo_solido
    Max = con_cuerpo_solido(MaxCaraV17)
    m = Max(glb=...)
    m.pose(clip, t)                          # se prepara solo la primera vez que ve cada clip (CUERPO_AUTO)
    m.cuerpo_solido(clip, cajas=[{"nombre": "mesa", "centro": [x, y, z], "medio": [hx, hy, hz]}])   # con objetos
    print(m.cuerpo_informe(clip))            # qué arregló y cuándo
Coordenadas de las cajas: las del render (metros, Z arriba). Se apaga con CUERPO_SOLIDO = False.
"""
import numpy as np

CZ = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
TRONCO = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Head']
CADENAS = {'brazo_L': ['LeftArm', 'LeftForeArm', 'LeftHand'], 'brazo_R': ['RightArm', 'RightForeArm', 'RightHand'],
           'pierna_L': ['LeftLeg', 'LeftShin', 'LeftFoot'], 'pierna_R': ['RightLeg', 'RightShin', 'RightFoot'], 'tronco': TRONCO}
CUERPOS = TRONCO + sum((v for k, v in CADENAS.items() if k != 'tronco'), [])
PADRE = {'Spine1': 'Hips', 'Spine2': 'Spine1', 'Chest': 'Spine2', 'Neck1': 'Chest', 'Head': 'Neck1'}
for _L in ('Left', 'Right'):
    PADRE.update({_L + 'Arm': 'Chest', _L + 'ForeArm': _L + 'Arm', _L + 'Hand': _L + 'ForeArm',
                  _L + 'Leg': 'Hips', _L + 'Shin': _L + 'Leg', _L + 'Foot': _L + 'Shin'})
PIEL = ('Std_Skin_Head', 'Std_Skin_Body', 'Std_Skin_Arm', 'Std_Skin_Leg', 'Sneaker', 'Std_Nails',
        'Pants')      # las piernas con el pantalón puesto: el brazo se apoya ENCIMA del pantalón, no dentro


# ---------------------------------------------------------------- cuaterniones (w, x, y, z), vectorizados
def q_de_R(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat()
    return np.concatenate([q[:, 3:], q[:, :3]], 1).reshape(R.shape[:-2] + (4,))

def R_de_q(q):
    from scipy.spatial.transform import Rotation
    q = np.asarray(q); f = q.reshape(-1, 4)
    return Rotation.from_quat(np.concatenate([f[:, 1:], f[:, :1]], 1)).as_matrix().reshape(q.shape[:-1] + (3, 3))

def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)

def qconj(a): return a * np.array([1, -1, -1, -1.])

def qnlerp(a, b, t):
    b = np.where((np.sum(a * b, -1, keepdims=True) < 0), -b, b)
    t = np.asarray(t)[..., None] if np.ndim(t) else t
    q = a * (1 - t) + b * t
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)

def qrotvec(q):
    q = np.where(q[..., :1] < 0, -q, q); v = q[..., 1:]; n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-9, v / np.maximum(n, 1e-12) * ang, 2 * v)


class _CuerpoSolido:
    CUERPO_SOLIDO = True
    CUERPO_AUTO = True          # se prepara solo la primera vez que se pide un clip
    CUERPO_FPS = 30             # cuadros por segundo de la simulación
    CUERPO_PISO = True          # el piso (z = 0) también es sólido
    CUERPO_HOLGURA = 0.008      # m que el brazo se queda separado del cuerpo
    CUERPO_UMBRAL = 0.006       # m que tiene que meterse un brazo para que se active la física
    CUERPO_UMBRAL_PISO = 0.025  # m bajo el piso / dentro de una caja para activarla (el motor ya cuida los pies)
    CUERPO_ANTES = 0.30         # s de entrada suave antes del choque
    CUERPO_DESPUES = 0.35       # s de salida suave después
    CUERPO_FREC = 18.0          # Hz: qué tan firme sigue el brazo a la animación
    CUERPO_TIRON = 0.25         # rad: tope del tirón hacia la animación (si la animación está muy adentro, no fuerza)
    CUERPO_SUAVE = 1.2          # cuadros: suavizado en el tiempo de la corrección
    CUERPO_VERBOSO = True

    # ------------------------------------------------------------------ API
    def cuerpo_solido(s, clip, cajas=(), piso=None):
        """Prepara (o vuelve a preparar) la física de un clip. Devuelve el informe."""
        c = s.__dict__.setdefault('_cs', {})
        c[id(clip)] = None                                   # mientras se prepara, la pose es la del motor
        c[id(clip)] = _preparar(s, clip, list(cajas), s.CUERPO_PISO if piso is None else piso)
        return c[id(clip)]['informe']

    def cuerpo_informe(s, clip):
        d = s.__dict__.get('_cs', {}).get(id(clip))
        return d['informe'] if d else None

    def globales(s, clip, t, *a, **k):
        G, Pm, b = super().globales(clip, t, *a, **k)
        if not s.CUERPO_SOLIDO:
            return G, Pm, b
        c = s.__dict__.setdefault('_cs', {})
        if id(clip) not in c:
            if not s.CUERPO_AUTO:
                return G, Pm, b
            s.cuerpo_solido(clip)
        d = c.get(id(clip))
        if d is None or not d['activo']:
            return G, Pm, b
        bucle = bool(a[0]) if a else bool(k.get('bucle', False))
        f = t * d['fps']
        if bucle and d['n'] > 1: f %= (d['n'] - 1)
        f = min(max(f, 0.0), d['n'] - 1.0); f0 = int(np.floor(f)); f1 = min(f0 + 1, d['n'] - 1); u = f - f0
        if not (d['hay'][f0] or d['hay'][f1]):
            return G, Pm, b
        Cq = qnlerp(d['C'][f0], d['C'][f1], u)                # corrección local de cada cuerpo del muñeco
        Ce = R_de_q(Cq); Cg = np.einsum('ji,njk,kl->nil', CZ, Ce, CZ)
        dp = (d['dp'][f0] * (1 - u) + d['dp'][f1] * u) @ CZ / s.base.scale     # CZ^T dp
        return _aplicar(s, np.array(G, float), np.array(Pm, float), Cg, dp, d) + (b,)


def _aplicar(s, G, Pm, Cg, dp, d):
    """FK otra vez con las correcciones locales: lo que no se corrige queda igual al milímetro."""
    pad = s.padres; ix = d['ix_cuerpo']; Gn = G.copy(); Pn = Pm.copy()
    for j in d['orden']:
        p = pad[j]
        if j == d['hips']:
            Gn[j] = G[j] @ Cg[0]; Pn[j] = Pm[j] + dp; continue
        L = G[p].T @ G[j]; o = G[p].T @ (Pm[j] - Pm[p])
        Gn[j] = Gn[p] @ L
        if ix[j] >= 0: Gn[j] = Gn[j] @ Cg[ix[j]]
        Pn[j] = Pn[p] + Gn[p] @ o
    return Gn, Pn


def _motor_a_fisica(s, G, Pm):
    """Pose del motor (glTF, Y arriba) -> rotaciones y posiciones en el espacio del render (Z arriba, metros)."""
    Re = np.einsum('ij,njk,lk->nil', CZ, G, CZ)
    Pe = (Pm @ CZ.T - np.array([0, 0, s.base.zoff])) * s.base.scale
    return Re, Pe


def _envolventes(s, b, Gs=None, Ps=None):
    """Por cada cuerpo del muñeco: puntos (en su marco local, metros) de la envolvente de la piel que mueve ese hueso."""
    from scipy.spatial import ConvexHull
    nom = s.nombres; pad = s.padres
    def cuerpo_de(j):
        while j >= 0 and nom[j] not in CUERPOS: j = pad[j]
        return nom[j] if j >= 0 else 'Hips'
    a_cuerpo = np.array([CUERPOS.index(cuerpo_de(j)) for j in range(len(nom))])
    punta = np.zeros(len(nom), bool)
    from soma.rig import _M
    muestras = [] if Gs is None else sorted(set(int(x) for x in np.linspace(0, len(Gs) - 1, 7)))
    Ks = [np.stack([_M(Gs[f][j], Ps[f][j]) @ b['ibm'][j] for j in range(len(nom))]) for f in muestras]
    pts = {n: [] for n in CUERPOS}
    for i, p in enumerate(s.prims):
        if not p['mname'].startswith(PIEL): continue
        J, W = s.Jn[i], s.Wn[i]; P = p['P'].astype(float)
        Wc = np.zeros((len(P), len(CUERPOS)))
        np.add.at(Wc, (np.repeat(np.arange(len(P)), J.shape[1]), a_cuerpo[J].ravel()), W.ravel())
        due = Wc.argmax(1); fuerte = (Wc.max(1) > 0.35) & ((W * punta[J]).sum(1) < 0.5)
        for k, n in enumerate(CUERPOS):
            sel = (due == k) & fuerte
            if sel.sum() < 4: continue
            j = s.idx[n]; Ph = np.c_[P[sel], np.ones(sel.sum())]
            if n.endswith('Hand') and Ks:                              # la mano, con los dedos de este clip
                for f, K in zip(muestras, Ks):
                    Mv = np.einsum('vk,vkij->vij', W[sel], K[J[sel]])
                    Vw = np.einsum('vij,vj->vi', Mv[:, :3, :], Ph)
                    pts[n].append((Vw - Ps[f][j]) @ Gs[f][j] @ CZ.T * s.base.scale)
                continue
            loc = (Ph @ np.asarray(b['ibm'][j]).T)[:, :3]              # al marco local del hueso
            pts[n].append(loc @ CZ.T * s.base.scale)
    out = {}
    for n, L in pts.items():
        if not L: continue
        X = np.concatenate(L)
        try: out[n] = X[ConvexHull(X).vertices]
        except Exception: out[n] = X
    return out


def _xml(s, env, off0, cajas, piso, excl, dt):
    def bits(n):
        if any(k in n for k in ('Arm', 'Hand')): return 'contype="1" conaffinity="1"', s.CUERPO_HOLGURA
        return 'contype="2" conaffinity="1"', 0.0
    hijos = {n: [h for h in CUERPOS if PADRE.get(h) == n] for n in CUERPOS}
    mallas = ''.join(f'<mesh name="h_{n}" vertex="{" ".join(f"{v:.5f}" for v in X.ravel())}"/>' for n, X in env.items())
    def cuerpo(n):
        o = off0[n]; art = '<freejoint name="raiz"/>' if n == 'Hips' else f'<joint name="{n}" type="ball"/>'
        bt, mg = bits(n)
        g = (f'<geom name="g_{n}" type="mesh" mesh="h_{n}" density="1000" friction="0.4 0.005 0.0001" margin="{mg}" solref="0.004 1" solimp="0.95 0.99 0.001" {bt}/>'
             if n in env else '<geom type="sphere" size="0.03" density="1000" contype="0" conaffinity="0"/>')
        return f'<body name="{n}" pos="{o[0]:.5f} {o[1]:.5f} {o[2]:.5f}">{art}{g}{"".join(cuerpo(h) for h in hijos[n])}</body>'
    ent = '<geom name="piso" type="plane" size="0 0 1" contype="4" conaffinity="3" friction="0.9 0.005 0.0001"/>' if piso else ''
    for k, c in enumerate(cajas):
        ce, h = c['centro'], np.abs(c['medio'])
        ent += (f'<geom name="caja{k}" type="box" size="{h[0]} {h[1]} {h[2]}" pos="{ce[0]} {ce[1]} {ce[2]}" '
                f'contype="4" conaffinity="3" friction="0.9 0.005 0.0001"/>')
    ex = ''.join(f'<exclude body1="{a}" body2="{b_}"/>' for a, b_ in sorted(excl))
    return (f'<mujoco><option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.81"/>'
            f'<asset>{mallas}</asset><worldbody>{ent}{cuerpo("Hips")}</worldbody><contact>{ex}</contact></mujoco>')


def _cadena_de(n):
    for k, v in CADENAS.items():
        if n in v: return k
    return None


def _preparar(s, clip, cajas, piso):
    import mujoco, time
    t0 = time.time(); fps = s.CUERPO_FPS; n = max(2, int(np.floor(clip.dur * fps + 1e-6)) + 1)
    sup = super(_CuerpoSolido, s).globales
    # 1. pose final del motor en cada cuadro
    Gs, Ps = [], []
    for f in range(n):
        G, Pm, b = sup(clip, min(f / fps, clip.dur))
        Gs.append(np.array(G, float)); Ps.append(np.array(Pm, float))
    Gs = np.array(Gs); Ps = np.array(Ps)
    Re = np.einsum('ij,fnjk,lk->fnil', CZ, Gs, CZ)
    Pe = (Ps @ CZ.T - np.array([0, 0, s.base.zoff])) * s.base.scale
    jx = np.array([s.idx[c_] for c_ in CUERPOS]); nb = len(CUERPOS)
    px = np.array([CUERPOS.index(PADRE[c_]) if c_ in PADRE else -1 for c_ in CUERPOS])
    Rb = Re[:, jx]; Pb = Pe[:, jx]                                         # (cuadros, cuerpos, ...)
    ql = np.zeros((n, nb, 4)); off = np.zeros((n, nb, 3))
    ql[:, 0] = q_de_R(Rb[:, 0]); off[:, 0] = Pb[:, 0]
    for k in range(1, nb):
        ql[:, k] = q_de_R(np.einsum('fji,fjk->fik', Rb[:, px[k]], Rb[:, k]))
        off[:, k] = np.einsum('fji,fj->fi', Rb[:, px[k]], Pb[:, k] - Pb[:, px[k]])
    for k in range(nb):                                                    # sin saltos de signo en el tiempo
        for f in range(1, n):
            if ql[f, k] @ ql[f - 1, k] < 0: ql[f, k] = -ql[f, k]
    # velocidades de referencia (angular en el marco local del hijo; lineal de la raíz en el mundo)
    wl = np.zeros((n, nb, 3)); wl[:-1] = qrotvec(qmul(qconj(ql[:-1]), ql[1:])) * fps; wl[-1] = wl[-2]
    vr = np.zeros((n, 3)); vr[:-1] = (off[1:, 0] - off[:-1, 0]) * fps; vr[-1] = vr[-2]
    # 2. muñeco con la forma real
    env = _envolventes(s, b, Gs, Ps)
    off0 = {c_: off[0, k] for k, c_ in enumerate(CUERPOS)}
    sub = max(1, int(round((1.0 / fps) / 0.002))); dt = (1.0 / fps) / sub
    excl = {tuple(sorted(p_)) for L in ('Left', 'Right') for p_ in ((L + 'Arm', 'Spine2'), (L + 'Arm', 'Neck1'))}
    m = mujoco.MjModel.from_xml_string(_xml(s, env, off0, cajas, piso, excl, dt)); d = mujoco.MjData(m)
    bid = np.array([m.body(c_).id for c_ in CUERPOS])
    # pares que ya se tocan en la pose de reposo de la malla (vecinos de verdad) -> no son choques
    Mb = [np.linalg.inv(np.asarray(b['ibm'][j])) for j in jx]
    Rr = np.array([CZ @ M_[:3, :3] @ CZ.T for M_ in Mb]); Pr = np.array([(CZ @ M_[:3, 3] - [0, 0, s.base.zoff]) * s.base.scale for M_ in Mb])
    qr = np.zeros((nb, 4)); orr = np.zeros((nb, 3)); qr[0] = q_de_R(Rr[0]); orr[0] = Pr[0]
    for k in range(1, nb):
        qr[k] = q_de_R(Rr[px[k]].T @ Rr[k]); orr[k] = Rr[px[k]].T @ (Pr[k] - Pr[px[k]])
    qadr = np.array([m.jnt_qposadr[m.body_jntadr[i]] for i in bid]); dadr = np.array([m.jnt_dofadr[m.body_jntadr[i]] for i in bid])
    def poner(q, o, w=None, v=None):
        d.qpos[0:3] = o[0]; d.qpos[3:7] = q[0]
        for k in range(1, nb):
            m.body_pos[bid[k]] = o[k]; d.qpos[qadr[k]:qadr[k] + 4] = q[k]
        if w is not None:
            d.qvel[0:3] = v; d.qvel[3:6] = w[0]
            for k in range(1, nb): d.qvel[dadr[k]:dadr[k] + 3] = w[k]
    poner(qr, orr); mujoco.mj_forward(m, d)
    gname = lambda g: m.geom(g).name
    for i in range(d.ncon):
        c_ = d.contact[i]; a_, b_ = gname(c_.geom1), gname(c_.geom2)
        if a_.startswith('g_') and b_.startswith('g_') and c_.dist < 0.002: excl.add(tuple(sorted((a_[2:], b_[2:]))))
    m = mujoco.MjModel.from_xml_string(_xml(s, env, off0, cajas, piso, excl, dt)); d = mujoco.MjData(m)
    bid = np.array([m.body(c_).id for c_ in CUERPOS])
    qadr = np.array([m.jnt_qposadr[m.body_jntadr[i]] for i in bid]); dadr = np.array([m.jnt_dofadr[m.body_jntadr[i]] for i in bid])
    gname = lambda g: m.geom(g).name
    cad = [_cadena_de(c_) for c_ in CUERPOS]; nomc = list(CADENAS); cix = np.array([nomc.index(c_) for c_ in cad])
    # 3. dónde hace falta: choques de la pose del motor
    falta = np.zeros((n, len(nomc)), bool); peor = np.zeros((n, len(nomc))); eventos = []
    def choques():
        out = []
        for i in range(d.ncon):
            c_ = d.contact[i]; a_, b_ = gname(c_.geom1), gname(c_.geom2)
            out.append((a_[2:] if a_.startswith('g_') else a_, b_[2:] if b_.startswith('g_') else b_, c_.dist))
        return out
    for f in range(n):
        poner(ql[f], off[f]); mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
        for a_, b_, dist in choques():
            ca, cb = _cadena_de(a_), _cadena_de(b_)
            if ca is None or cb is None:                                   # con el piso o una caja
                if dist > -s.CUERPO_UMBRAL_PISO: continue
                cc = ca or cb; quien = [cc] if cc != 'tronco' else nomc
            else:
                if dist > -s.CUERPO_UMBRAL: continue
                quien = [c__ for c__ in (ca, cb) if c__.startswith('brazo')]
            for q_ in quien:
                k = nomc.index(q_); falta[f, k] = True
                if -dist > peor[f, k]: peor[f, k] = -dist
            eventos.append((f, a_, b_, -dist))
    informe = dict(eventos=eventos, excl=sorted(excl), clip=getattr(clip, 'ruta', ''), cuadros=n, fps=fps, ventanas=[], antes_cm=round(float(peor.max()) * 100, 1), despues_cm=0.0)
    d_out = dict(fps=fps, n=n, activo=False, informe=informe)
    if not falta.any():
        if s.CUERPO_VERBOSO: print(f'[cuerpo sólido] sin choques ({n} cuadros, {time.time() - t0:.1f} s)')
        return d_out
    # peso de cada cadena en el tiempo: 1 en el choque, con entrada y salida suaves
    na, nd = int(round(s.CUERPO_ANTES * fps)), int(round(s.CUERPO_DESPUES * fps))
    w = np.zeros((n, len(nomc)))
    for k in range(len(nomc)):
        idx = np.where(falta[:, k])[0]
        for f in range(n):
            if not len(idx): break
            antes = idx[idx >= f]; desp = idx[idx <= f]
            a1 = 1 - (antes[0] - f) / max(na, 1) if len(antes) else 0
            a2 = 1 - (f - desp[-1]) / max(nd, 1) if len(desp) else 0
            x = np.clip(max(a1, a2), 0, 1); w[f, k] = x * x * (3 - 2 * x)
    wb = w[:, cix]                                                          # por cuerpo
    for k, nm in enumerate(nomc):
        idx = np.where(w[:, k] > 0)[0]
        if len(idx):
            cortes = np.where(np.diff(idx) > 1)[0]
            for a_, b_ in zip(np.r_[idx[0], idx[cortes + 1]], np.r_[idx[cortes], idx[-1]]):
                informe['ventanas'].append(dict(cadena=nm, desde=round(a_ / fps, 2), hasta=round(b_ / fps, 2),
                                                cm=round(float(peor[a_:b_ + 1, k].max()) * 100, 1)))
    # 4. ganancias (frecuencia natural fija sobre la inercia de cada subárbol)
    poner(ql[0], off[0]); mujoco.mj_forward(m, d)
    kp = np.zeros(nb); kd = np.zeros(nb); wn = 2 * np.pi * s.CUERPO_FREC
    for k in range(1, nb):
        i = bid[k]; L = np.linalg.norm(d.subtree_com[i] - d.xpos[i]); I = m.body_subtreemass[i] * L * L + m.body_inertia[i].max() + 0.002
        kp[k] = wn * wn * I; kd[k] = 2 * 0.9 * np.sqrt(kp[k] * I)
        m.dof_damping[dadr[k]:dadr[k] + 3] = kd[k]; m.dof_armature[dadr[k]:dadr[k] + 3] = max(0.02, 0.05 * I)
    arm0 = m.dof_armature.copy(); PESADO = 1e5
    Mtot = m.body_subtreemass[bid[0]]; kr = 40000.0
    # 5. simulación: solo corre donde algo hace falta; lo demás va prescrito por la animación
    qs = ql.copy(); ps = off[:, 0].copy(); activo = (w > 0).any(1); f = 0; restos = []
    while f < n - 1:
        if not (activo[f] or activo[f + 1]):
            f += 1; continue
        poner(ql[f], off[f], wl[f], vr[f]); mujoco.mj_forward(m, d)
        while f < n - 1 and (activo[f] or activo[f + 1]):
            libre = np.maximum(wb[f], wb[f + 1]) > 0                     # lo que no está libre no cede ante un choque
            m.dof_armature[0:6] = arm0[0:6] if libre[0] else PESADO
            for k in range(1, nb): m.dof_armature[dadr[k]:dadr[k] + 3] = arm0[dadr[k]:dadr[k] + 3] if libre[k] else PESADO
            for kk in range(sub):
                u = (kk + 1) / sub; u0 = kk / sub
                qt = qnlerp(ql[f], ql[f + 1], u0); ot = off[f] * (1 - u0) + off[f + 1] * u0
                qn = qnlerp(ql[f], ql[f + 1], u); on = off[f] * (1 - u) + off[f + 1] * u
                wt = wb[f] * (1 - u0) + wb[f + 1] * u0; wnx = wb[f] * (1 - u) + wb[f + 1] * u
                for k in range(1, nb): m.body_pos[bid[k]] = ot[k]
                d.qfrc_applied[:] = 0
                cur = np.array([d.qpos[qadr[k]:qadr[k] + 4] for k in range(1, nb)])
                err = qrotvec(qmul(qconj(cur), qt[1:]))
                ne = np.linalg.norm(err, axis=1, keepdims=True); err = err * np.minimum(1.0, s.CUERPO_TIRON / np.maximum(ne, 1e-9))
                for k in range(1, nb):
                    if wt[k] > 0: d.qfrc_applied[dadr[k]:dadr[k] + 3] = kp[k] * err[k - 1] + kd[k] * wl[f, k]
                if wt[0] > 0:                                              # tronco libre: titiritero suave
                    F = kr * (ot[0] - d.qpos[0:3]) - 2 * np.sqrt(kr * Mtot) * (d.qvel[0:3] - vr[f]) + Mtot * np.array([0, 0, 9.81]) * 0.7
                    e = qrotvec(qmul(qconj(d.qpos[3:7].copy()), qt[0]))
                    d.qfrc_applied[0:3] = F; d.qfrc_applied[3:6] = kr * 0.15 * e - 2 * np.sqrt(kr * 0.15 * 2.0) * (d.qvel[3:6] - wl[f, 0])
                mujoco.mj_step(m, d)
                # lo que no está en física vuelve a la animación (del todo si w = 0; en parte durante la entrada/salida)
                for k in range(nb):
                    a_ = 1 - wnx[k]
                    if a_ <= 0: continue
                    if k == 0:
                        d.qpos[0:3] = d.qpos[0:3] * (1 - a_) + on[0] * a_; d.qpos[3:7] = qnlerp(d.qpos[3:7].copy(), qn[0], a_)
                        d.qvel[0:3] = d.qvel[0:3] * (1 - a_) + vr[f] * a_; d.qvel[3:6] = d.qvel[3:6] * (1 - a_) + wl[f, 0] * a_
                    else:
                        d.qpos[qadr[k]:qadr[k] + 4] = qnlerp(d.qpos[qadr[k]:qadr[k] + 4].copy(), qn[k], a_)
                        d.qvel[dadr[k]:dadr[k] + 3] = d.qvel[dadr[k]:dadr[k] + 3] * (1 - a_) + wl[f, k] * a_
            f += 1
            qs[f, 0] = d.qpos[3:7]; ps[f] = d.qpos[0:3]
            for k in range(1, nb): qs[f, k] = d.qpos[qadr[k]:qadr[k] + 4]
            for k in range(1, nb): m.body_pos[bid[k]] = off[f, k]
            mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
            for a_, b_, dist in choques():
                ca, cb = _cadena_de(a_), _cadena_de(b_)
                if ca and cb and dist < 0: restos.append(-dist); informe.setdefault('restos', []).append((f, a_, b_, round(-dist * 100, 1)))
    C = qmul(qconj(ql), qs); C = np.where(C[..., :1] < 0, -C, C)
    dp = ps - off[:, 0]
    if s.CUERPO_SUAVE > 0:                                   # quita el temblor fino del choque (no cambia la pose)
        from scipy.ndimage import gaussian_filter1d
        C = gaussian_filter1d(C, s.CUERPO_SUAVE, axis=0, mode='nearest'); C /= np.linalg.norm(C, axis=-1, keepdims=True)
        dp = gaussian_filter1d(dp, s.CUERPO_SUAVE, axis=0, mode='nearest')
    hay = (np.abs(C[..., 1:]).max((1, 2)) > 1e-6) | (np.abs(dp).max(1) > 1e-6)
    ixc = np.full(len(s.nombres), -1); ixc[jx] = np.arange(nb)
    hips = s.idx['Hips']
    orden = [j for j in range(len(s.nombres)) if _desciende(s, j, hips)]
    informe['despues_cm'] = round(float(max(restos)) * 100, 1) if restos else 0.0
    informe['segundos'] = round(time.time() - t0, 1)
    d_out.update(activo=True, C=C, dp=dp, hay=hay, ix_cuerpo=ixc, hips=hips, orden=orden)
    if s.CUERPO_VERBOSO:
        print(f"[cuerpo sólido] {len(informe['ventanas'])} tramos corregidos; se metía hasta {informe['antes_cm']} cm, "
              f"queda {informe['despues_cm']} cm ({informe['segundos']} s)")
    return d_out


def _desciende(s, j, raiz):
    while j >= 0:
        if j == raiz: return True
        j = s.padres[j]
    return False


def con_cuerpo_solido(Clase):
    """La misma clase de Max con el cuerpo sólido como última capa de pose (no modifica la clase original)."""
    return type(Clase.__name__ + 'Solido', (_CuerpoSolido, Clase), {})
