"""Lector GLB (glTF 2.0 binario) estático: mallas, UV, normales, textura base y jerarquía de nodos."""
import json, struct, io, numpy as np
from PIL import Image
CT={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def q2m(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def node_matrix(n):
    if 'matrix' in n: return np.array(n['matrix'],float).reshape(4,4).T
    M=np.eye(4); M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0]); return M
def load(path):
    d=open(path,'rb').read(); assert d[:4]==b'glTF'
    o=12; J=None; B=b''
    while o<len(d):
        ln,ty=struct.unpack('<II',d[o:o+8]); ch=d[o+8:o+8+ln]; o+=8+ln
        if ty==0x4E4F534A: J=json.loads(ch)
        elif ty==0x004E4942: B=ch
    def bv(i): v=J['bufferViews'][i]; s=v.get('byteOffset',0); return B[s:s+v['byteLength']],v.get('byteStride')
    def acc(i):
        a=J['accessors'][i]; raw,stride=bv(a['bufferView']); off=a.get('byteOffset',0); dt=CT[a['componentType']]; nc=NC[a['type']]
        isz=np.dtype(dt).itemsize*nc
        if stride and stride!=isz:
            arr=np.frombuffer(raw,np.uint8)[off:off+stride*a['count']].reshape(a['count'],stride)[:,:isz].copy().view(dt).reshape(a['count'],nc)
        else: arr=np.frombuffer(raw,dt,count=a['count']*nc,offset=off).reshape(a['count'],nc)
        arr=arr.astype(np.float32)
        if a.get('normalized'): arr/= np.iinfo(dt).max
        return arr
    imgs={}
    def image(ti):
        tex=J['textures'][ti]; si=tex.get('source')
        if si is None: return None
        if si not in imgs:
            im=J['images'][si]
            if 'bufferView' in im: imgs[si]=Image.open(io.BytesIO(bv(im['bufferView'])[0])).convert('RGBA')
            else: imgs[si]=None
        return imgs[si]
    parts=[]
    def walk(ni,P):
        n=J['nodes'][ni]; M=P@node_matrix(n)
        if 'mesh' in n:
            for pr in J['meshes'][n['mesh']]['primitives']:
                at=pr['attributes']; pos=acc(at['POSITION']); nor=acc(at['NORMAL']) if 'NORMAL' in at else None; uv=acc(at['TEXCOORD_0']) if 'TEXCOORD_0' in at else np.zeros((len(pos),2),np.float32)
                idx=acc(pr['indices']).astype(int).ravel() if 'indices' in pr else np.arange(len(pos))
                if pr.get('mode',4)!=4: continue
                Pw=(np.c_[pos,np.ones(len(pos))]@M.T)[:,:3]
                Nw=(nor@np.linalg.inv(M[:3,:3]).T) if nor is not None else None
                mat=J['materials'][pr['material']] if 'material' in pr else {}
                pbr=mat.get('pbrMetallicRoughness',{}); col=pbr.get('baseColorFactor',[1,1,1,1]); im=None
                if 'baseColorTexture' in pbr: im=image(pbr['baseColorTexture']['index'])
                parts.append(dict(pos=Pw,nor=Nw,uv=uv,idx=idx,color=col,image=im,alpha=mat.get('alphaMode','OPAQUE'),name=mat.get('name','')))
        for c in n.get('children',[]): walk(c,M)
    sc=J['scenes'][J.get('scene',0)]
    for ni in sc['nodes']: walk(ni,np.eye(4))
    return parts,J

class Skinned:
    """glTF con piel y animaciones (caballo). Salida Z-arriba, escala a altura dada."""
    def __init__(s, path, height=None):
        d=open(path,'rb').read(); o=12; s.J=None; s.B=b''
        while o<len(d):
            ln,ty=struct.unpack('<II',d[o:o+8]); ch=d[o+8:o+8+ln]; o+=8+ln
            if ty==0x4E4F534A: s.J=json.loads(ch)
            elif ty==0x004E4942: s.B=ch
        J=s.J; s.parent={}
        for i,n in enumerate(J['nodes']):
            for c in n.get('children',[]): s.parent[c]=i
        s.rest=[(np.array(n.get('translation',[0,0,0]),float),np.array(n.get('rotation',[0,0,0,1]),float),np.array(n.get('scale',[1,1,1]),float),n.get('matrix')) for n in J['nodes']]
        sk=J['skins'][0]; s.joints=sk['joints']; s.ibm=s.acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        s.anims={a['name']:a for a in J.get('animations',[])}
        s.prims=[]; s.images={}
        for ni,n in enumerate(J['nodes']):
            if 'mesh' not in n: continue
            for pr in J['meshes'][n['mesh']]['primitives']:
                at=pr['attributes']; P=s.acc(at['POSITION']); N=s.acc(at['NORMAL']); UV=s.acc(at['TEXCOORD_0']); idx=s.acc(pr['indices']).astype(int).ravel()
                Jn=s.acc(at['JOINTS_0']).astype(int); W=s.acc(at['WEIGHTS_0']); W/=np.maximum(W.sum(1,keepdims=True),1e-9)
                mat=J['materials'][pr['material']] if 'material' in pr else {}; im=None
                pbr=mat.get('pbrMetallicRoughness',{})
                if 'baseColorTexture' in pbr:
                    ti=J['textures'][pbr['baseColorTexture']['index']]['source']; iv=J['images'][ti]
                    raw=s.view(iv['bufferView']); im=Image.open(io.BytesIO(raw)).convert('RGBA')
                s.prims.append(dict(P=P,N=N,UV=UV,idx=idx,J=Jn,W=W,image=im,color=pbr.get('baseColorFactor',[1,1,1,1]),targets=pr.get('targets',[]),tnames=J['meshes'][n['mesh']].get('extras',{}).get('targetNames',[]),alpha=mat.get('alphaMode','OPAQUE'),mname=mat.get('name','')))
        s.scale=1.0; s.zoff=0
        P=s.pose(None,0,apply=False); z=np.concatenate([p[0] for p in P]); s.zoff=z[:,2].min()
        if height: s.scale=height/(z[:,2].max()-z[:,2].min())
    def view(s,i):
        v=s.J['bufferViews'][i]; o=v.get('byteOffset',0); return s.B[o:o+v['byteLength']]
    def acc(s,i):
        a=s.J['accessors'][i]; nc0=NC[a['type']]
        if 'bufferView' not in a:
            arr=np.zeros((a['count'],nc0),np.float32)
        else: arr=s._acc_dense(a)
        if 'sparse' in a:
            sp=a['sparse']; n=sp['count']; iv=s.J['bufferViews'][sp['indices']['bufferView']]; it=CT[sp['indices']['componentType']]
            idx=np.frombuffer(s.B[iv.get('byteOffset',0)+sp['indices'].get('byteOffset',0):],it,count=n).astype(int)
            vv=s.J['bufferViews'][sp['values']['bufferView']]; vt=CT[a['componentType']]
            vals=np.frombuffer(s.B[vv.get('byteOffset',0)+sp['values'].get('byteOffset',0):],vt,count=n*nc0).reshape(n,nc0).astype(np.float32)
            arr=arr.copy(); arr[idx]=vals
        return arr
    def _acc_dense(s,a):
        v=s.J['bufferViews'][a['bufferView']]; raw=s.B[v.get('byteOffset',0)+a.get('byteOffset',0):]
        dt=CT[a['componentType']]; nc=NC[a['type']]; st=v.get('byteStride'); isz=np.dtype(dt).itemsize*nc
        if st and st!=isz: arr=np.frombuffer(raw[:st*a['count']],np.uint8).reshape(a['count'],st)[:,:isz].copy().view(dt).reshape(a['count'],nc)
        else: arr=np.frombuffer(raw,dt,count=a['count']*nc).reshape(a['count'],nc)
        arr=arr.astype(np.float32)
        if a.get('normalized'): arr/=np.iinfo(dt).max
        return arr
    def local(s,i,trs):
        T,R,S,M=trs
        if M is not None and i not in (trs and []): pass
        if s.rest[i][3] is not None and not s._anim_touch.get(i): return np.array(s.rest[i][3],float).reshape(4,4).T
        m=np.eye(4); m[:3,:3]=q2m(R)@np.diag(S); m[:3,3]=T; return m
    def sample(s,name,t):
        a=s.anims[name]; out={}; ln=0
        for ch in a['channels']:
            sm=a['samplers'][ch['sampler']]; ti=s.acc(sm['input']).ravel(); ln=max(ln,ti[-1])
        tt=t%ln if ln>0 else 0
        for ch in a['channels']:
            sm=a['samplers'][ch['sampler']]; ti=s.acc(sm['input']).ravel(); vo=s.acc(sm['output']); node=ch['target']['node']; path=ch['target']['path']
            k=np.searchsorted(ti,tt)
            if k<=0: v=vo[0]
            elif k>=len(ti): v=vo[-1]
            else:
                u=(tt-ti[k-1])/max(ti[k]-ti[k-1],1e-9); a0,b0=vo[k-1],vo[k]
                if path=='rotation':
                    if np.dot(a0,b0)<0: b0=-b0
                    v=a0*(1-u)+b0*u; v/=np.linalg.norm(v)
                else: v=a0*(1-u)+b0*u
            out.setdefault(node,{})[path]=v
        return out,ln
    def set_morph(s,weights):
        """weights: {nombre_blendshape: peso 0..1}. Precalcula posiciones con las formas aplicadas."""
        if not hasattr(s,'_tcache'): s._tcache={}
        for pi,p in enumerate(s.prims):
            P=p['P'].astype(np.float64).copy()
            for name,w in weights.items():
                if w==0 or name not in p['tnames']: continue
                k=p['tnames'].index(name)
                if k>=len(p['targets']) or 'POSITION' not in p['targets'][k]: continue
                key=(pi,k)
                if key not in s._tcache: s._tcache[key]=s.acc(p['targets'][k]['POSITION'])
                P+=w*s._tcache[key]
            p['Pm']=P
    def pose(s,anim,t,apply=True,local_override=None):
        ov=s.sample(anim,t)[0] if anim else {}; s._anim_touch=ov; G={}; LO=local_override or {}
        def g(i):
            if i in G: return G[i]
            T,R,S,M=s.rest[i]; o=ov.get(i,{})
            if i in LO: L=LO[i]
            elif M is not None and not o: L=np.array(M,float).reshape(4,4).T
            else:
                L=np.eye(4); L[:3,:3]=q2m(o.get('rotation',R))@np.diag(o.get('scale',S)); L[:3,3]=o.get('translation',T)
            G[i]=(g(s.parent[i])@L) if i in s.parent else L; return G[i]
        K=np.stack([g(j)@s.ibm[k] for k,j in enumerate(s.joints)])
        res=[]; Cz=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
        for p in s.prims:
            Mv=np.einsum('vk,vkij->vij',p['W'],K[p['J']])
            V=np.einsum('vij,vj->vi',Mv[:,:3,:3],p.get('Pm',p['P']))+Mv[:,:3,3]; Nn=np.einsum('vij,vj->vi',Mv[:,:3,:3],p['N'])
            V=V@Cz.T; Nn=Nn@Cz.T
            if apply: V=(V-np.array([0,0,s.zoff]))*s.scale
            Nn/=np.maximum(np.linalg.norm(Nn,axis=1,keepdims=True),1e-9)
            res.append((V[p['idx']],Nn[p['idx']]))
        s.G=G
        return res
    def joint_pos(s,name_part):
        for k,j in enumerate(s.joints):
            if name_part.lower() in s.J['nodes'][j].get('name','').lower():
                p=s.G[j][:3,3]; Cz=np.array([[1,0,0],[0,0,-1],[0,1,0.]]); return ((Cz@p)-[0,0,s.zoff])*s.scale
