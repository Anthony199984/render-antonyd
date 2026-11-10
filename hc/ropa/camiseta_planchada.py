"""CAMISETA PLANCHADA: última pasada sobre la camiseta, en cada cuadro, para que se comporte como tela:
 1. CEDE AL BRAZO: donde el brazo aprieta la camiseta (colgando pegado al cuerpo, cruzado sobre la barriga...), la tela
    se hunde hacia el cuerpo en vez de que el brazo la atraviese. Las mangas y la axila no se tocan.
 2. AGACHADO O SENTADO (FALDA LIMPIA): la parte de abajo de la camiseta cuelga lisa, como de pie, y lo que queda dentro
    de los muslos se esconde justo debajo de la tela del pantalón: se ve la camiseta metida en el pliegue del regazo,
    sin picos ni pedazos, y nada asoma por debajo de los muslos. (Antes se intentaba subirla punto por punto y el borde
    quedaba roto: por eso aquí se apagan CAMISETA_SENTADO y el empuje del pantalón mientras está agachado.)
 3. NO SE ROMPE: de pie, ningún pedazo queda estirado en pico.
 4. EL FORRO (la capa de adentro) va siempre pegado por dentro de la de afuera: no asoma.
Solo cambia lo que está mal: de pie y sin tocar nada, la camiseta queda igual que antes."""
import numpy as np


def _lis(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


class _CamisetaPlanchada:
    CAMISETA_PLANCHADA = True
    CAMISETA_SENTADO = False     # (capa vieja del motor) la reemplaza la FALDA LIMPIA de aquí
    FALDA_DESDE, FALDA_HASTA = 30.0, 55.0     # grados de muslo levantado: empieza / del todo
    FALDA_BAJO = 0.012           # m: lo escondido queda así de adentro de la cara de arriba del muslo

    def _tela_sobre_piel(s, k, V, piel, margen):
        c = getattr(s, '_falda_c', 0.0) if s.CAMISETA_PLANCHADA else 0.0
        if c <= 0.0:
            return super()._tela_sobre_piel(k, V, piel, margen)
        sin = [(g, v) for g, v in piel if g != 'pantalon']       # agachado: el pantalón ya no empuja la camiseta
        V2 = super()._tela_sobre_piel(k, V, sin, margen)
        if c >= 1.0:
            return V2
        return super()._tela_sobre_piel(k, V, piel, margen) * (1 - c) + V2 * c
    PLANCHA_VUELTAS = 6
    PLANCHA_ESTIRA = 1.35        # veces su largo: más que esto, la tela se recoge
    PLANCHA_MARGEN = 0.004       # m por fuera del cuerpo / pantalón
    PLANCHA_BRAZO = 0.006        # m entre el brazo y la tela que aprieta
    FORRO_ADENTRO = 0.0025       # m del forro por dentro de la tela de afuera

    def _plancha_prep(s):
        c = s.__dict__.get('_plancha')
        if c is not None:
            return c
        from scipy.spatial import cKDTree
        ko = next((i for i, p in enumerate(s.prims) if p['mname'].startswith('Poloche') and 'Interior' not in p['mname']), None)
        ki = next((i for i, p in enumerate(s.prims) if p['mname'].startswith('Poloche') and 'Interior' in p['mname']), None)
        if ko is None:
            s._plancha = {}; return s._plancha
        u = 1.0 / s.base.scale; c = dict(ko=ko, ki=ki)
        for k in (ko, ki):
            if k is None: continue
            d = s._malla_tela(k); q = s._prep_axila(k); P0 = s.prims[k]['P'].astype(float)[d['rep']]
            manga = np.maximum(q['manga']['Left'], q['manga']['Right'])
            yh = min(s.pA['LeftArm'][1], s.pA['RightArm'][1])
            a = _lis((yh - 0.02 * u - P0[:, 1]) / (0.08 * u)) * (1 - _lis((manga - 0.15) / 0.3))
            F = d['F']; E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
            L0 = np.linalg.norm(P0[E[:, 0]] - P0[E[:, 1]], axis=1) * s.base.scale
            ok = L0 > 1e-5
            ds = np.minimum(np.linalg.norm(P0 - s.pA['LeftArm'], axis=1), np.linalg.norm(P0 - s.pA['RightArm'], axis=1)) * s.base.scale
            ab = a * (manga < 0.02) * _lis((ds - 0.11) / 0.07)          # lejos del hombro y nada de manga
            c[k] = dict(a=a, ab=ab, E=E[ok], L0=L0[ok], P0=P0, val=np.maximum(np.bincount(E[ok].ravel(), minlength=len(P0)), 1))
        if ki is not None:                                     # cada punto del forro con su punto de la tela de afuera
            dist, nn = cKDTree(c[ko]['P0']).query(c[ki]['P0'])
            c[ki]['par'] = nn; c[ki]['cerca'] = dist * s.base.scale < 0.012
        c['cuerpo'] = [i for i, p in enumerate(s.prims) if p['mname'].startswith('Std_Skin_Body')]
        c['pant'] = [i for i, p in enumerate(s.prims) if p['mname'].startswith('Pants')]
        s._plancha = c
        return c

    def _plancha_capsulas(s):
        """Brazos como cápsulas (inicio, fin, radio al inicio, radio al final), en metros del render."""
        J = s.joint_pos; out = []
        for L in ('Left', 'Right'):
            A_, B_ = J(L + 'Arm'), J(L + 'ForeArm')
            out.append((A_ + 0.45 * (B_ - A_), B_, 0.050, 0.047))        # del brazo, solo la mitad de abajo
            out.append((J(L + 'ForeArm'), J(L + 'Hand'), 0.046, 0.032))
            fin = J(L + 'HandMiddle2') if (L + 'HandMiddle2') in s.idx else J(L + 'Hand')
            out.append((J(L + 'Hand'), fin, 0.034, 0.028))
        return out

    def _plancha_piel(s, G, Pm, b, lista):
        from soma.rig import _M
        K = np.stack([_M(G[j], Pm[j]) @ b['ibm'][j] for j in range(len(G))])
        Cz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]]); Vs, Ns = [], []
        for i in lista:
            p = s.prims[i]; Mv = np.einsum('vk,vkij->vij', s.Wn[i], K[s.Jn[i]])
            V = np.einsum('vij,vj->vi', Mv[:, :3, :3], p.get('Pm', p['P'])) + Mv[:, :3, 3]
            Vs.append((V @ Cz.T - [0, 0, s.base.zoff]) * s.base.scale); Ns.append(np.einsum('vij,vj->vi', Mv[:, :3, :3], p['N']) @ Cz.T)
        V = np.concatenate(Vs); N = np.concatenate(Ns)
        return V, N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _por_vertice(par, idx, n):
        V, N = par; X = np.zeros((n, 3)); X[idx] = V; Nn = np.zeros((n, 3)); Nn[idx] = N
        return X, Nn

    def pose_mats(s, G, Pm, b, apply=True):
        ang = s._muslo_arriba(Pm) if hasattr(s, '_muslo_arriba') else {}
        s._falda_c = float(_lis((max(ang.values(), default=0.0) - s.FALDA_DESDE) / (s.FALDA_HASTA - s.FALDA_DESDE)))
        res = super().pose_mats(G, Pm, b, apply)
        if not (s.CAMISETA_PLANCHADA and apply):
            return res
        c = s._plancha_prep()
        if not c or not len(res[c['ko']][0]):
            return res
        from scipy.spatial import cKDTree
        ko, ki = c['ko'], c['ki']; q = c[ko]; d = s._malla_tela(ko); p = s.prims[ko]
        Vv, Nv = s._por_vertice(res[ko], p['idx'], len(p['P']))
        X0 = Vv[d['rep']]; X = X0.copy(); a = q['a']; act = a > 0.01
        # lo sólido: cuerpo (debajo de la camiseta) y pantalón (solo sus caras de arriba y de los lados)
        cols = [s._plancha_piel(G, Pm, b, c['cuerpo'])] if c['cuerpo'] else []
        for i in c['pant']:
            if len(res[i][0]):
                Pv, Pn = res[i]; sel = Pn[:, 2] > -0.3
                cols.append((Pv[sel][::2], Pn[sel][::2]))
        CP = np.concatenate([x for x, _ in cols]); CN = np.concatenate([n_ for _, n_ in cols]); arbol = cKDTree(CP)
        caps = s._plancha_capsulas()
        eje0 = s.joint_pos('Hips'); eje1 = s.joint_pos('Head'); ev = eje1 - eje0; el = ev @ ev
        E, L0, S = q['E'], q['L0'], d['S']

        def afuera(X, sel):
            dist, j = arbol.query(X[sel]); sd = np.einsum('ij,ij->i', X[sel] - CP[j], CN[j])
            f = np.where((dist < 0.07) & (sd < s.PLANCHA_MARGEN) & (sd > -0.09), s.PLANCHA_MARGEN - sd, 0.0)
            return f[:, None] * CN[j]

        def brazos(X, sel):
            """Empuje hacia el cuerpo de la tela que quedó dentro de un brazo."""
            Xs = X[sel]; D = np.zeros_like(Xs)
            tt = np.clip(((Xs - eje0) @ ev) / el, 0, 1); hacia = eje0 + tt[:, None] * ev - Xs
            hacia /= np.maximum(np.linalg.norm(hacia, axis=1, keepdims=True), 1e-9)
            for A_, B_, r0, r1 in caps:
                e = B_ - A_; ln = np.linalg.norm(e)
                if ln < 1e-4: continue
                e = e / ln; h = (Xs - A_) @ e; hc = np.clip(h, 0, ln)
                R = r0 + (r1 - r0) * hc / ln + s.PLANCHA_BRAZO
                w = (Xs - A_) - np.outer(h, e)
                fuera_eje = np.maximum(np.abs(h - hc), 0)                       # pasado de las puntas (media esfera)
                dentro = (w * w).sum(1) + fuera_eje ** 2 < R * R
                if not dentro.any(): continue
                Rr = np.sqrt(np.maximum(R * R - fuera_eje ** 2, 1e-8))
                dp = hacia - np.outer(hacia @ e, e); A2 = (dp * dp).sum(1); Bq = (w * dp).sum(1); Cq = (w * w).sum(1) - Rr * Rr
                disc = np.maximum(Bq * Bq - A2 * Cq, 0); t = np.where(A2 > 1e-6, (-Bq + np.sqrt(disc)) / np.maximum(A2, 1e-6), 0.0)
                t = np.clip(t, 0, 2 * R)
                D += np.where(dentro[:, None], hacia * t[:, None], 0.0)
            return D

        tocado = np.zeros(len(X), bool)
        ab = q['ab']; actb = ab > 0.01; cuanto = s._falda_c
        escondido = np.zeros(len(X), bool)
        # 0. FALDA LIMPIA: lo que queda dentro de un muslo o debajo de él se recoge hacia el centro del muslo (escondido)
        if cuanto > 0.02:
            tops = [(res[i][0][res[i][1][:, 2] > 0.35]) for i in c['pant'] if len(res[i][0])]
            T = np.concatenate(tops) if tops else np.zeros((0, 3))
            if len(T) > 50:
                a2 = cKDTree(T[:, :2]); idx_act = np.where(act)[0]
                dd, jj = a2.query(X[idx_act, :2], k=8, distance_upper_bound=0.035)
                zt = np.where(np.isfinite(dd), T[np.minimum(jj, len(T) - 1), 2], -9.0).max(1)
                z = X[idx_act, 2]; col = _lis((0.035 - dd[:, 0]) / 0.02)           # qué tan metido en la sombra del muslo
                bajo = (zt > -8) & (z < zt - s.FALDA_BAJO)
                if bajo.any():
                    sel = idx_act[bajo]; Xs = X[sel]; mejor = None
                    for L in ('Left', 'Right'):                                # punto más cercano del eje cadera-rodilla
                        A_, B_ = s.joint_pos(L + 'Leg'), s.joint_pos(L + 'Shin'); e = B_ - A_
                        tt = np.clip(((Xs - A_) @ e) / (e @ e), 0.0, 0.92); Pn = A_ + tt[:, None] * e
                        dn = np.linalg.norm((Xs - Pn)[:, :2], axis=1)
                        if mejor is None: mejor = (dn, Pn)
                        else: mejor = (np.minimum(mejor[0], dn), np.where((dn < mejor[0])[:, None], Pn, mejor[1]))
                    w = (cuanto * a[sel] * col[bajo])[:, None]
                    Dz = np.zeros_like(X); Dz[sel] = w * (mejor[1] - Xs); mov0 = np.zeros(len(X), bool); mov0[sel] = w[:, 0] > 0.5
                    Dv = Dz.copy()
                    for _ in range(3): Dv = 0.5 * Dv + 0.5 * (S @ Dv)              # los vecinos acompañan: borde sin picos
                    Dz = np.where(mov0[:, None], Dz, Dv * a[:, None])
                    X += Dz; escondido |= mov0; tocado |= np.abs(Dz).sum(1) > 2e-4
        # sentado en el piso: la falda que quedaría enterrada se recoge dentro de la cadera (no se aplasta en el piso)
        if cuanto > 0.02:
            hondo = _lis((-X[:, 2] - 0.01) / 0.04) * cuanto * a
            if (hondo > 0.01).any():
                cad = s.joint_pos('Hips'); obj = cad + 0.35 * (X - cad); obj[:, 2] = np.maximum(obj[:, 2], 0.03)
                X += hondo[:, None] * (obj - X); escondido |= hondo > 0.3; tocado |= hondo > 0.01
        X[:, 2] = np.maximum(X[:, 2], 0.004)                                   # nunca debajo del piso
        Xf = X.copy()                                                          # la falda ya puesta: de aquí no se mueve
        for vuelta in range(s.PLANCHA_VUELTAS):
            Dn = np.zeros_like(X)
            # 1. cede al brazo
            Db = brazos(X, actb); Dn[actb] += Db * ab[actb, None]
            # 2. no se estira de más (cada punta de un borde estirado se acerca a la otra)
            v = X[E[:, 1]] - X[E[:, 0]]; L = np.linalg.norm(v, axis=1); lim = s.PLANCHA_ESTIRA * L0
            mal = (L > lim) & ~escondido[E[:, 0]] & ~escondido[E[:, 1]]
            if mal.any() and cuanto < 0.5:
                cor = (v[mal] * ((L[mal] - lim[mal]) / L[mal])[:, None]) * 0.5
                Ds = np.zeros_like(X); np.add.at(Ds, E[mal, 0], cor); np.add.at(Ds, E[mal, 1], -cor)
                cnt = np.zeros(len(X)); np.add.at(cnt, E[mal].ravel(), 1)
                Dn += Ds / np.maximum(cnt, 1)[:, None]
            mov = (np.abs(Dn).sum(1) > 1e-6)
            if not mov.any() and vuelta > 0: break
            tocado |= mov
            X = X + Dn * a[:, None]
            # 3. el cambio se reparte con los vecinos (hundido suave, sin escalón)
            D = X - Xf
            for _ in range(2):
                D = np.where(mov[:, None], 0.5 * D + 0.5 * (S @ D), 0.5 * (S @ D))
            D = D * a[:, None] * ~escondido[:, None]; tocado |= np.abs(D).sum(1) > 2e-4
            X = Xf + D
            # 4. por fuera del cuerpo y del pantalón (lo escondido no: va por dentro a propósito)
            sel = act & tocado & ~escondido
            if sel.any(): X[sel] += afuera(X, sel) * a[sel, None]
        if not tocado.any() and ki is None:
            return res
        D = X - X0
        if tocado.any():
            Vn = Vv + D[d['inv']]
            Nn = s._normales(ko, Vn, Nv)
            wN = np.clip(np.linalg.norm(D, axis=1) / 0.01, 0, 1)[d['inv']][:, None]
            Nn = Nv * (1 - wN) + Nn * wN; Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
            res[ko] = (Vn[p['idx']], Nn[p['idx']])
        else:
            Vn, Nn = Vv, Nv
        # el forro: pegado por dentro de la tela de afuera (donde no es manga)
        if ki is not None and len(res[ki][0]):
            qi = c[ki]; di = s._malla_tela(ki); pi = s.prims[ki]
            Vi, Ni = s._por_vertice(res[ki], pi['idx'], len(pi['P']))
            Xo = Vn[d['rep']]; No = Nn[d['rep']]
            Xi = Vi[di['rep']]; obj = Xo[qi['par']] - No[qi['par']] * s.FORRO_ADENTRO
            wi = (qi['a'] * qi['cerca'])[:, None]
            Xi2 = Xi * (1 - wi) + obj * wi
            Vi2 = Vi + (Xi2 - Xi)[di['inv']]
            res[ki] = (Vi2[pi['idx']], Ni[pi['idx']])
        return res
