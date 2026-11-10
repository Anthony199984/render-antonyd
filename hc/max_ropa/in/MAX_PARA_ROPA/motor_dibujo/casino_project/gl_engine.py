"""Small, reproducible OpenGL 3D renderer. Geometry and textures are procedural.
No frames, textures, models, or sound from the reference video are used here.
Requires Linux Mesa EGL/OpenGL, NumPy, Pillow. No downloaded assets.
"""
import os
os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
import ctypes as C
import math
import numpy as np
from PIL import Image

def normalize(a):
    a=np.asarray(a,dtype=np.float32)
    return a/max(float(np.linalg.norm(a)),1e-8)

def matrix(pos=(0,0,0),scale=(1,1,1),rot=(0,0,0)):
    x,y,z=rot
    cx,sx,cy,sy,cz,sz=math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
    rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],np.float32)
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],np.float32)
    rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],np.float32)
    a=np.eye(4,dtype=np.float32);a[:3,:3]=rz@ry@rx@np.diag(scale);a[:3,3]=pos
    return a

def look_at(eye,target,up=(0,0,1)):
    e=np.asarray(eye,np.float32);f=normalize(np.asarray(target)-e);s=normalize(np.cross(f,up));u=np.cross(s,f)
    a=np.eye(4,dtype=np.float32);a[:3,:3]=[s,u,-f];a[:3,3]=-a[:3,:3]@e
    return a

def perspective(fov,aspect,near=.035,far=80):
    f=1/math.tan(math.radians(fov)/2)
    return np.array([[f/aspect,0,0,0],[0,f,0,0],[0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]],np.float32)

def ortho(l,r,b,t,n,f):
    return np.array([[2/(r-l),0,0,-(r+l)/(r-l)],[0,2/(t-b),0,-(t+b)/(t-b)],[0,0,-2/(f-n),-(f+n)/(f-n)],[0,0,0,1]],np.float32)

def tri_mesh(triangles,uvs=None,colors=None,smooth_normals=None):
    a=[]
    for i,tri in enumerate(triangles):
        tri=np.asarray(tri,np.float32);n=normalize(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
        for j,p in enumerate(tri):
            uv=(0,0) if uvs is None else uvs[i][j]
            col=(1,1,1) if colors is None else colors[i][j]
            nn=n if smooth_normals is None else smooth_normals[i][j]
            a.append([*p,*nn,*uv,*col])
    return np.asarray(a,np.float32)

def box():
    v=np.array([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]])
    faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    ts=[];uv=[]
    for f in faces:
        for ids in [(0,1,2),(0,2,3)]:
            ts.append(v[[f[j] for j in ids]]);uv.append([[(0,0),(1,0),(1,1),(0,1)][j] for j in ids])
    return tri_mesh(ts,uv)

def plane():
    return tri_mesh([[[-.5,-.5,0],[.5,-.5,0],[.5,.5,0]],[[-.5,-.5,0],[.5,.5,0],[-.5,.5,0]]],[[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]])

def sphere(n=20,m=14):
    ts=[];uv=[];nn=[]
    def point(i,j):
        a=2*math.pi*i/n;b=math.pi*j/m
        return np.array([math.sin(b)*math.cos(a),math.sin(b)*math.sin(a),math.cos(b)])
    for j in range(m):
        for i in range(n):
            ps=[point(i,j),point(i+1,j),point(i+1,j+1),point(i,j+1)]
            us=[(i/n,1-j/m),((i+1)/n,1-j/m),((i+1)/n,1-(j+1)/m),(i/n,1-(j+1)/m)]
            for ids in [(0,2,1),(0,3,2)]:
                ts.append([ps[k] for k in ids]);nn.append([ps[k] for k in ids]);uv.append([us[k] for k in ids])
    return tri_mesh(ts,uv,smooth_normals=nn)

def cylinder(n=20,r1=1,r2=1):
    ts=[];uv=[];nn=[]
    for i in range(n):
        a=2*math.pi*i/n;b=2*math.pi*(i+1)/n
        p=[(r1*math.cos(a),r1*math.sin(a),-.5),(r1*math.cos(b),r1*math.sin(b),-.5),(r2*math.cos(b),r2*math.sin(b),.5),(r2*math.cos(a),r2*math.sin(a),.5)]
        na=normalize((math.cos(a),math.sin(a),r1-r2));nb=normalize((math.cos(b),math.sin(b),r1-r2));normals=[na,nb,nb,na]
        for ids in [(0,1,2),(0,2,3)]:
            ts.append([p[k] for k in ids]);uv.append([[(i/n,0),((i+1)/n,0),((i+1)/n,1),(i/n,1)][k] for k in ids]);nn.append([normals[k] for k in ids])
        ts.extend([[(0,0,-.5),p[1],p[0]],[(0,0,.5),p[3],p[2]]]);uv.extend([[(.5,.5),((math.cos(b)+1)/2,(math.sin(b)+1)/2),((math.cos(a)+1)/2,(math.sin(a)+1)/2)],[(.5,.5),((math.cos(a)+1)/2,(math.sin(a)+1)/2),((math.cos(b)+1)/2,(math.sin(b)+1)/2)]]);nn.extend([[(0,0,-1)]*3,[(0,0,1)]*3])
    return tri_mesh(ts,uv,smooth_normals=nn)

VERTEX='''#version 330 core
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV; layout(location=3) in vec3 aColor;
uniform mat4 uModel,uVP,uLightVP; uniform mat3 uNormal;
out vec3 wPos, normal, color;out vec2 uv;out vec4 lightPos;
void main(){ vec4 p=uModel*vec4(aPos,1); wPos=p.xyz;normal=normalize(uNormal*aNormal);uv=aUV;color=aColor;lightPos=uLightVP*p;gl_Position=uVP*p;}
'''
FRAGMENT='''#version 330 core
in vec3 wPos,normal,color;in vec2 uv;in vec4 lightPos;
uniform sampler2D uTex,uShadow;uniform vec4 uColor;uniform vec3 uEye,uP0,uP1,uC0,uC1,uSun;
uniform float uAmbient,uEmission,uRoughness,uTexScale,uFog,uShadows,uBump,uSkin,uExposure,uVignette,uSunPower,uSpot,uPS1;
uniform vec3 uFire,uFireColor,uFogColor;uniform float uFogMax;uniform vec2 uResolution,uRepeat;
out vec4 frag;
float shadow(){vec3 p=lightPos.xyz/lightPos.w*.5+.5;if(p.z>1.||p.x<0.||p.x>1.||p.y<0.||p.y>1.)return 1.;float s=0.;vec2 texel=1./textureSize(uShadow,0);float bias=.0015;for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++){float z=texture(uShadow,p.xy+vec2(x,y)*texel*1.5).r;s+=p.z-bias>z?.27:1.;}return mix(1.,s/9.,uShadows);}
vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
float lum(vec3 x){return dot(x,vec3(.2126,.7152,.0722));}
vec3 bumpNormal(vec3 n){
 vec2 coord=uv*uRepeat;vec2 ds=1./vec2(textureSize(uTex,0));
 float h=lum(texture(uTex,coord).rgb),hx=lum(texture(uTex,coord+vec2(ds.x,0)).rgb),hy=lum(texture(uTex,coord+vec2(0,ds.y)).rgb);
 vec3 p1=dFdx(wPos),p2=dFdy(wPos);vec2 t1=dFdx(coord),t2=dFdy(coord);
 vec3 a=cross(p2,n),b=cross(n,p1);vec3 tangent=a*t1.x+b*t2.x,bitangent=a*t1.y+b*t2.y;
 float scale=inversesqrt(max(max(dot(tangent,tangent),dot(bitangent,bitangent)),.000001));
 return normalize(n-(hx-h)*uBump*tangent*scale-(hy-h)*uBump*bitangent*scale);
}
void main(){
 vec4 tx=texture(uTex,uv*uRepeat);float alpha=tx.a*uColor.a;if(alpha<.015)discard;
 vec3 n=normalize(normal);if(!gl_FrontFacing)n=-n;if(uBump>0.)n=bumpNormal(n);
 vec3 base=pow(max(tx.rgb*color*uColor.rgb,vec3(0.)),vec3(2.2));vec3 v=normalize(uEye-wPos);vec3 sun=normalize(uSun);float sh=shadow();
 vec3 illum=vec3(uAmbient)+vec3(.96,.94,.90)*uSunPower*max(0.,dot(n,sun))*sh;
 vec3 l0=uP0-wPos,l1=uP1-wPos;float d0=length(l0),d1=length(l1);l0=normalize(l0);l1=normalize(l1);
 float spot=mix(1.,smoothstep(-.04,.70,l0.z),uSpot);
 float wrapped0=mix(max(0.,dot(n,l0)),max(0.,(dot(n,l0)+.22)/1.22),uSkin);
 float wrapped1=mix(max(0.,dot(n,l1)),max(0.,(dot(n,l1)+.15)/1.15),uSkin);
 vec3 i0=uC0*wrapped0*spot/(1.+.14*d0*d0);vec3 i1=uC1*wrapped1/(1.+.20*d1*d1);
 illum+=i0*mix(1.,sh,.78)+i1*mix(1.,sh,.30);
 // Face-only diffuse bounce approximation, driven by the actual scene lights.
 // It vanishes when those lights are off; it is not material self-emission.
 illum+=uSkin*.10*(uC0/(1.+.14*d0*d0)+uC1/(1.+.20*d1*d1))*mix(.55,1.,sh);
 vec3 fire=uFire-wPos;float fd=length(fire);illum+=uFireColor*max(0.,dot(n,normalize(fire)))/(1.+24.*fd*fd);
 float power=mix(88.,12.,uRoughness);float s0=pow(max(0.,dot(n,normalize(l0+v))),power),s1=pow(max(0.,dot(n,normalize(l1+v))),power);
 float fresnel=.04+.26*pow(1.-max(0.,dot(n,v)),5.);vec3 spec=(uC0*s0/(1.+.14*d0*d0)+uC1*s1/(1.+.20*d1*d1))*fresnel*(1.-uRoughness)*1.2;
 vec3 rgb=base*illum+spec+base*uEmission;
 float fog=clamp(1.-exp(-length(uEye-wPos)*uFog*1.6),0.,uFogMax);rgb=mix(rgb,uFogColor,fog);
 rgb=pow(aces(rgb*uExposure),vec3(1./2.2));vec2 pp=gl_FragCoord.xy/uResolution;float vv=pow(length((pp-.5)*vec2(1.1,1.)),2.);
 rgb*=1.-uVignette*vv;if(uPS1>.5){float d=mod(floor(gl_FragCoord.x)+2.*floor(gl_FragCoord.y),4.)/4.-.375;rgb=floor(clamp(rgb+d/64.,0.,1.)*31.+.5)/31.;}frag=vec4(clamp(rgb,0.,1.),alpha);
}
'''
DEPTH_VERTEX='''#version 330 core
layout(location=0) in vec3 aPos;layout(location=2) in vec2 aUV;uniform mat4 uModel,uVP;out vec2 uv;void main(){uv=aUV;gl_Position=uVP*uModel*vec4(aPos,1);}
'''
DEPTH_FRAGMENT='''#version 330 core
in vec2 uv;uniform sampler2D uTex;uniform float uTexScale;void main(){if(texture(uTex,uv*uTexScale).a<.4)discard;}
'''

class Renderer:
    def __init__(self,width,height):
        self.w,self.h=width,height;self.e=C.CDLL('libEGL.so.1');self.g=C.CDLL('libGL.so.1');self.meshes={};self.textures={};self.mesh_data={}
        e=self.e
        def ef(name,ret,args):
            f=getattr(e,name);f.restype=ret;f.argtypes=args;return f
        get=ef('eglGetDisplay',C.c_void_p,[C.c_void_p]);ini=ef('eglInitialize',C.c_uint,[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)])
        choose=ef('eglChooseConfig',C.c_uint,[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_void_p),C.c_int,C.POINTER(C.c_int)])
        bind=ef('eglBindAPI',C.c_uint,[C.c_uint]);surf=ef('eglCreatePbufferSurface',C.c_void_p,[C.c_void_p,C.c_void_p,C.POINTER(C.c_int)])
        ctx=ef('eglCreateContext',C.c_void_p,[C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_int)]);current=ef('eglMakeCurrent',C.c_uint,[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p])
        self.display=get(None);a=C.c_int();b=C.c_int();assert ini(self.display,C.byref(a),C.byref(b));assert bind(0x30A2)
        conf=C.c_void_p();count=C.c_int();attrs=(C.c_int*15)(0x3033,1,0x3040,8,0x3024,8,0x3023,8,0x3022,8,0x3021,8,0x3025,24,0x3038)
        assert choose(self.display,attrs,C.byref(conf),1,C.byref(count))
        self.surface=surf(self.display,conf,(C.c_int*5)(0x3057,width,0x3056,height,0x3038));self.context=ctx(self.display,conf,None,(C.c_int*1)(0x3038));assert current(self.display,self.surface,self.surface,self.context)
        self.bind_gl();self.program=self.program_from(VERTEX,FRAGMENT);self.depth_program=self.program_from(DEPTH_VERTEX,DEPTH_FRAGMENT)
        self.glEnable(0x0B71);self.glEnable(0x0BE2);self.glBlendFunc(0x0302,0x0303);self.glPixelStorei(0x0CF5,1);self.glPixelStorei(0x0D05,1)
        self.add_texture('white',Image.new('RGBA',(2,2),(255,255,255,255)))
        self.shadow_size=2048;self.shadow=C.c_uint();self.glGenTextures(1,C.byref(self.shadow));self.glBindTexture(0x0DE1,self.shadow.value)
        self.glTexImage2D(0x0DE1,0,0x81A6,self.shadow_size,self.shadow_size,0,0x1902,0x1406,None)
        for p,v in [(0x2801,0x2601),(0x2800,0x2601),(0x2802,0x812F),(0x2803,0x812F)]:self.glTexParameteri(0x0DE1,p,v)
        self.fbo=C.c_uint();self.glGenFramebuffers(1,C.byref(self.fbo));self.glBindFramebuffer(0x8D40,self.fbo.value);self.glFramebufferTexture2D(0x8D40,0x8D00,0x0DE1,self.shadow.value,0);self.glDrawBuffer(0);self.glReadBuffer(0);assert self.glCheckFramebufferStatus(0x8D40)==0x8CD5;self.glBindFramebuffer(0x8D40,0)
        self.lightvp=ortho(-8,8,-8,8,.1,25)@look_at((2,-4,10),(0,1,0))
        self.pixels=np.empty((self.h,self.w,3),np.uint8)

    def bind_gl(self):
        U=C.c_uint;I=C.c_int;F=C.c_float;P=C.c_void_p
        funcs={'glEnable':(None,[U]),'glDisable':(None,[U]),'glBlendFunc':(None,[U,U]),'glDepthMask':(None,[C.c_ubyte]),'glPixelStorei':(None,[U,I]),'glViewport':(None,[I,I,I,I]),'glClearColor':(None,[F,F,F,F]),'glClear':(None,[U]),'glCreateShader':(U,[U]),'glShaderSource':(None,[U,I,C.POINTER(C.c_char_p),C.POINTER(I)]),'glCompileShader':(None,[U]),'glGetShaderiv':(None,[U,U,C.POINTER(I)]),'glGetShaderInfoLog':(None,[U,I,C.POINTER(I),C.c_char_p]),'glCreateProgram':(U,[]),'glAttachShader':(None,[U,U]),'glLinkProgram':(None,[U]),'glGetProgramiv':(None,[U,U,C.POINTER(I)]),'glUseProgram':(None,[U]),'glGetUniformLocation':(I,[U,C.c_char_p]),'glUniformMatrix4fv':(None,[I,I,C.c_ubyte,C.POINTER(F)]),'glUniformMatrix3fv':(None,[I,I,C.c_ubyte,C.POINTER(F)]),'glUniform3f':(None,[I,F,F,F]),'glUniform4f':(None,[I,F,F,F,F]),'glUniform1f':(None,[I,F]),'glUniform1i':(None,[I,I]),'glGenBuffers':(None,[I,C.POINTER(U)]),'glBindBuffer':(None,[U,U]),'glBufferData':(None,[U,C.c_size_t,P,U]),'glGenVertexArrays':(None,[I,C.POINTER(U)]),'glBindVertexArray':(None,[U]),'glEnableVertexAttribArray':(None,[U]),'glVertexAttribPointer':(None,[U,I,U,C.c_ubyte,I,P]),'glDrawArrays':(None,[U,I,I]),'glGenTextures':(None,[I,C.POINTER(U)]),'glActiveTexture':(None,[U]),'glBindTexture':(None,[U,U]),'glTexImage2D':(None,[U,I,I,I,I,I,U,U,P]),'glTexParameteri':(None,[U,U,I]),'glGenerateMipmap':(None,[U]),'glReadPixels':(None,[I,I,I,I,U,U,P]),'glGenFramebuffers':(None,[I,C.POINTER(U)]),'glBindFramebuffer':(None,[U,U]),'glFramebufferTexture2D':(None,[U,U,U,U,I]),'glCheckFramebufferStatus':(U,[U]),'glDrawBuffer':(None,[U]),'glReadBuffer':(None,[U]),'glFinish':(None,[])}
        for name,(ret,args) in funcs.items():
            f=getattr(self.g,name);f.restype=ret;f.argtypes=args;setattr(self,name,f)
        self.glUniform2f=self.g.glUniform2f;self.glUniform2f.restype=None;self.glUniform2f.argtypes=[I,F,F]
        self.locations={}

    def program_from(self,vert,frag):
        shaders=[]
        for st,code in [(0x8B31,vert),(0x8B30,frag)]:
            s=self.glCreateShader(st);data=C.c_char_p(code.encode());self.glShaderSource(s,1,C.byref(data),None);self.glCompileShader(s);ok=C.c_int();self.glGetShaderiv(s,0x8B81,C.byref(ok))
            if not ok.value:
                buf=C.create_string_buffer(4096);self.glGetShaderInfoLog(s,4096,None,buf);raise RuntimeError(buf.value.decode())
            shaders.append(s)
        p=self.glCreateProgram()
        for s in shaders:self.glAttachShader(p,s)
        self.glLinkProgram(p);ok=C.c_int();self.glGetProgramiv(p,0x8B82,C.byref(ok));assert ok.value
        return p

    def loc(self,p,name):
        k=(p,name)
        if k not in self.locations:self.locations[k]=self.glGetUniformLocation(p,name.encode())
        return self.locations[k]
    def mat(self,p,name,m):
        m=np.ascontiguousarray(np.asarray(m,np.float32).T);self.glUniformMatrix4fv(self.loc(p,name),1,0,m.ctypes.data_as(C.POINTER(C.c_float)))
    def scalar(self,p,name,v):self.glUniform1f(self.loc(p,name),v)
    def vec(self,p,name,v):self.glUniform3f(self.loc(p,name),*v)
    def add_mesh(self,name,data):
        data=np.ascontiguousarray(data,np.float32);vbo=C.c_uint();vao=C.c_uint();self.glGenBuffers(1,C.byref(vbo));self.glGenVertexArrays(1,C.byref(vao));self.glBindVertexArray(vao.value);self.glBindBuffer(0x8892,vbo.value);self.glBufferData(0x8892,data.nbytes,data.ctypes.data,0x88E4)
        for i,size,offset in [(0,3,0),(1,3,12),(2,2,24),(3,3,32)]:self.glEnableVertexAttribArray(i);self.glVertexAttribPointer(i,size,0x1406,0,44,C.c_void_p(offset))
        self.meshes[name]=(vao.value,len(data))
        self.mesh_data[name]=data

    def batch_static(self,objects,prefix):
        groups={};kept=[]
        for o in objects:
            # Keep doorway trim individually addressable for the reveal cut.
            if -.6<float(o['model'][1,3])<-.2 and o['mesh']=='box':kept.append(o);continue
            key=tuple((k,str(v)) for k,v in sorted(o.items()) if k not in ['mesh','model'])
            groups.setdefault(key,[]).append(o)
        result=[]
        for i,group in enumerate(groups.values()):
            if len(group)==1:result.extend(group);continue
            data=[]
            for o in group:
                a=self.mesh_data[o['mesh']].copy();m=o['model'];a[:,:3]=a[:,:3]@m[:3,:3].T+m[:3,3]
                a[:,3:6]=a[:,3:6]@np.linalg.inv(m[:3,:3]);a[:,3:6]/=np.maximum(np.linalg.norm(a[:,3:6],axis=1,keepdims=True),1e-8);data.append(a)
            name=f'{prefix}_{i}';self.add_mesh(name,np.concatenate(data));ob=dict(group[0]);ob['mesh']=name;ob['model']=np.eye(4,dtype=np.float32);result.append(ob)
        return result+kept
    def add_texture(self,name,image,repeat=True):
        a=np.ascontiguousarray(np.array(image.convert('RGBA'))[::-1]);t=C.c_uint();self.glGenTextures(1,C.byref(t));self.glBindTexture(0x0DE1,t.value);self.glTexImage2D(0x0DE1,0,0x8058,a.shape[1],a.shape[0],0,0x1908,0x1401,a.ctypes.data)
        for p,v in [(0x2801,0x2703),(0x2800,0x2601),(0x2802,0x2901 if repeat else 0x812F),(0x2803,0x2901 if repeat else 0x812F)]:self.glTexParameteri(0x0DE1,p,v)
        self.glGenerateMipmap(0x0DE1)
        if name.startswith(('ps1_','skinpart_')) or name in ['wood','deskwood','carpet','wall','ceiling','chair_fabric']:
            self.glTexParameteri(0x0DE1,0x2801,0x2600);self.glTexParameteri(0x0DE1,0x2800,0x2600)
        self.textures[name]=t.value

    def set_texture_filter(self,name,kind):
        """Per-texture sampling; existing set/body defaults stay unchanged."""
        self.glBindTexture(0x0DE1,self.textures[name])
        self.glTexParameteri(0x0DE1,0x2800,0x2600 if kind=='nearest' else 0x2601)
        self.glTexParameteri(0x0DE1,0x2801,0x2600 if kind=='nearest' else 0x2703)

    def draw(self,objects,camera,lighting,shadow=True):
        eye,target,fov=camera;vp=perspective(fov,self.w/self.h)@look_at(eye,target)
        if 'shadow_eye' in lighting:
            size=lighting.get('shadow_extent',6);self.lightvp=ortho(-size,size,-size,size,.10,28)@look_at(lighting['shadow_eye'],lighting.get('shadow_target',(0,0,0)))
        self.glBindFramebuffer(0x8D40,self.fbo.value);self.glViewport(0,0,self.shadow_size,self.shadow_size);self.glClear(0x0100);self.glUseProgram(self.depth_program);self.mat(self.depth_program,'uVP',self.lightvp);self.glActiveTexture(0x84C0)
        if shadow:
            for o in objects:
                if o.get('emission',0)>0.4 or o.get('transparent'):continue
                self.object(o,self.depth_program,True)
        self.glBindFramebuffer(0x8D40,0);self.glViewport(0,0,self.w,self.h);self.glClearColor(*lighting.get('background',(.02,.018,.016)),1);self.glClear(0x4100);p=self.program;self.glUseProgram(p);self.mat(p,'uVP',vp);self.mat(p,'uLightVP',self.lightvp);self.vec(p,'uEye',eye)
        for name,key,default in [('uP0','p0',(0,-.5,3)),('uP1','p1',(3,-3,3)),('uC0','c0',(1.2,.88,.55)),('uC1','c1',(.22,.25,.32)),('uSun','sun',(.3,-.4,1))]:self.vec(p,name,lighting.get(key,default))
        self.scalar(p,'uAmbient',lighting.get('ambient',.24));self.scalar(p,'uFog',lighting.get('fog',0));self.vec(p,'uFogColor',lighting.get('fogcolor',(.12,.135,.145)));self.scalar(p,'uFogMax',lighting.get('fogmax',.6));self.scalar(p,'uShadows',1 if shadow else 0)
        for name,key,default in [('uExposure','exposure',1),('uVignette','vignette',.38),('uSunPower','sunpower',.25),('uSpot','spot',0)]:self.scalar(p,name,lighting.get(key,default))
        self.vec(p,'uFire',lighting.get('fire',(0,0,-10)));self.vec(p,'uFireColor',lighting.get('firecolor',(0,0,0)));self.glUniform2f(self.loc(p,'uResolution'),self.w,self.h)
        self.glActiveTexture(0x84C1);self.glBindTexture(0x0DE1,self.shadow.value);self.glUniform1i(self.loc(p,'uShadow'),1);self.glUniform1i(self.loc(p,'uTex'),0)
        for o in objects:
            if not o.get('transparent'):self.object(o,p)
        self.glDepthMask(0)
        transparent=[o for o in objects if o.get('transparent')]
        transparent.sort(key=lambda o:-np.linalg.norm(o['model'][:3,3]-np.asarray(eye)))
        for o in transparent:self.object(o,p)
        self.glDepthMask(1);self.glReadPixels(0,0,self.w,self.h,0x1907,0x1401,self.pixels.ctypes.data);return self.pixels[::-1].copy()

    def object(self,o,p,depth=False):
        self.mat(p,'uModel',o['model']);self.glActiveTexture(0x84C0);self.glBindTexture(0x0DE1,self.textures[o.get('texture','white')]);self.scalar(p,'uTexScale',o.get('texscale',1))
        if not depth:
            repeat=o.get('texrepeat',(o.get('texscale',1),)*2);self.glUniform2f(self.loc(p,'uRepeat'),*repeat)
            nm=np.ascontiguousarray(np.linalg.inv(o['model'][:3,:3]).astype(np.float32));self.glUniformMatrix3fv(self.loc(p,'uNormal'),1,0,nm.ctypes.data_as(C.POINTER(C.c_float)))
            col=o.get('color',(1,1,1));self.glUniform4f(self.loc(p,'uColor'),*col[:3],col[3] if len(col)>3 else 1);self.scalar(p,'uEmission',o.get('emission',0));self.scalar(p,'uRoughness',o.get('roughness',.8));self.scalar(p,'uBump',o.get('bump',0));self.scalar(p,'uSkin',o.get('skin',0));self.scalar(p,'uPS1',o.get('ps1',0))
        vao,count=self.meshes[o['mesh']];self.glBindVertexArray(vao);self.glDrawArrays(0x0004,0,count)
