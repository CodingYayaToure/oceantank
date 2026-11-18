uniform float uT; uniform vec2 uC; uniform float uI; uniform float uRot; uniform float uR; uniform vec2 uZ; uniform float uBurst; uniform float uDis;
const float D=3.2; const float HW=4.0;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec2 hash2(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return fract(sin(p)*43758.5453);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<4;i++){s+=a*vnoise(p);p=p*2.03+17.1;a*=.5;}return s;}
float caus(vec2 p,float t){vec2 n=floor(p),f=fract(p);float d1=8.,d2=8.;
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){vec2 g=vec2(float(i),float(j));vec2 o=.5+.5*sin(t+6.2831*hash2(n+g));
    float d=length(g+o-f);if(d<d1){d2=d1;d1=d;}else if(d<d2){d2=d;}}
  return pow(1.-smoothstep(0.,.28,d2-d1),3.);}
float warm(float x){return smoothstep(uZ.x-2.6,uZ.x+1.6,x)*(1.-smoothstep(uZ.y-1.6,uZ.y+2.6,x));}
float cold(float x){return smoothstep(7.6,11.4,x);}
float amb(vec2 p){float h=0.;
  h+=.075*sin(p.x*.9+uT*1.1+sin(p.y*.6+uT*.4));
  h+=.05*sin(p.y*1.3-uT*1.3+p.x*.4);
  h+=.035*sin((p.x+p.y)*2.1+uT*1.9);
  h+=.03*sin((p.x*1.7-p.y*2.6)*1.3-uT*2.4);
  return h;}
float ring(vec2 d){float r=length(d);return sin(3.4*r-5.6*uT)*smoothstep(1.6*uR,3.*uR,r)*exp(-r/8.5);}
float H(vec2 p){
  vec2 d=p-uC;float r=length(d)+1e-4;
  float back=max(uC.x-p.x,0.);
  float wake=smoothstep(0.,2.,back)*exp(-back/8.)*exp(-pow(d.y/(1.8+.22*back),2.));
  float h=amb(p)*(1.+uI*(1.5*exp(-r*r/30.)+1.6*wake));
  float th=atan(d.y,d.x);float lr=log(r/uR+.15);
  h+=uI*(-1.75*exp(-pow(r/(.6*uR),2.))+.26*exp(-pow((r-1.1*uR)/(.5*uR),2.))+.2*exp(-pow(r/(3.2*uR),2.)));
  h+=uI*.2*sin(2.*(th+uRot)-5.*lr)*smoothstep(.9*uR,1.8*uR,r)*exp(-r/(3.2*uR));
  h+=uI*(.2+.4*uBurst)*(ring(d)+.55*ring(vec2(d.x,2.*HW-p.y-uC.y))+.55*ring(vec2(d.x,-2.*HW-p.y-uC.y)));
  return h;}varying vec3 vP;
void main(){vec3 p=position;p.y=D+H(p.xz);vP=p;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);}
