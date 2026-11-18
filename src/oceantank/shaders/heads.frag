varying float vA;void main(){vec2 c=gl_PointCoord-.5;float d=dot(c,c);if(d>.25)discard;gl_FragColor=vec4(vec3(.9,.97,1.),vA*smoothstep(.25,0.,d));}
