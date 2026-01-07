import numpy
import math
import h5py

with h5py.File("dynamicivbody.ombvh5", "r") as f:
  Nsp = (f["ivobject"]["data"].shape[1]-1)//6
Ncs = 200

nsp = numpy.zeros((Ncs,3))
border = numpy.zeros((Ncs,), dtype=int)
r=0.1
i=0
nsp[i]=numpy.array([0,0,2*r]); border[i]=True; i+=1
nsp[i]=numpy.array([0,0,2*0]); border[i]=True; i+=1
nsp[i]=numpy.array([r,0,2*0]); border[i]=True; i+=1
da=math.pi/2/(Ncs-2);
for a in numpy.arange(da, math.pi/2-da/2, da):
  nsp[i]=numpy.array([r*math.cos(a),0,2*r*math.sin(a)]); border[i]=False; i+=1
nspStr=""
for r in range(0,nsp.shape[0]):
  nspStr+=f"vec3({','.join(map(lambda x: str(x), nsp[r]))}),\n"
nspStr=nspStr[:-2]
borderStr=",".join(map(lambda x: str(x), border))

normal = numpy.zeros((2*Ncs,3))
for csIdx in range(0,Ncs):
  nIdx = 2*csIdx
  normal[nIdx+1] = numpy.array([
    +(nsp[csIdx+(1 if csIdx<Ncs-1 else 1-Ncs)][2] - nsp[csIdx][2]),
    0,
    -(nsp[csIdx+(1 if csIdx<Ncs-1 else 1-Ncs)][0] - nsp[csIdx][0])
  ])
  normal[nIdx+1] = normal[nIdx+1]/numpy.linalg.norm(normal[nIdx+1]);
  normal[nIdx+(2 if csIdx<Ncs-1 else 2-2*Ncs)] = normal[nIdx+1];
for csIdx in range(0,Ncs):
  # combine normals
  nIdx = 2*csIdx
  if border[csIdx]==0:
    n1 = normal[nIdx+1]
    n2 = normal[nIdx]
    n1 = n1 + n2
    normal[nIdx+1] = n1/numpy.linalg.norm(n1)
    normal[nIdx] = n1
normalStr=""
for r in range(0,normal.shape[0]):
  normalStr+=f"vec3({','.join(map(lambda x: str(x), normal[r]))}),\n"
normalStr=normalStr[:-2]

vertexAndNormal = numpy.zeros((2*Nsp*Ncs,3))

meshCoordIndex = numpy.zeros(((Nsp-1)*Ncs*5,), dtype=int)
i=0
for spIdx in range(0,Nsp-1):
  for csIdx in range(0,Ncs):
    nIdx = 2*(spIdx*Ncs+csIdx)
    meshCoordIndex[i] = nIdx+1; i+=1
    meshCoordIndex[i] = nIdx+(2 if csIdx<Ncs-1 else 2-2*Ncs); i+=1
    meshCoordIndex[i] = nIdx+(2 if csIdx<Ncs-1 else 2-2*Ncs)+2*Ncs; i+=1
    meshCoordIndex[i] = nIdx+1+2*Ncs; i+=1
    meshCoordIndex[i] = -1; i+=1

Nline=numpy.sum(border)
tubeCoordIndex = numpy.zeros((Nline*(Nsp+1),), dtype=int)
i=0
for csIdx in range(0,Ncs):
  if border[csIdx]==0:
    continue
  for spIdx in range(0,Nsp):
    nIdx = 2*(spIdx*Ncs+csIdx)
    tubeCoordIndex[i] = nIdx; i+=1
  tubeCoordIndex[i] = -1; i+=1

bbox = [0,6.28  ,  -0.5,0.5  ,  0.1,0.2]



content=f'''#Inventor V2.1 ascii

Separator {{
  # bbox
  Separator {{
    PickStyle {{
      style UNPICKABLE
    }}
    Material {{
      transparency 1
    }}
    Translation {{
      translation { (bbox[0]+bbox[1])/2 } { (bbox[2]+bbox[3])/2 } { (bbox[4]+bbox[5])/2 }
    }}
    Cube {{
      width  { bbox[1]-bbox[0] }
      height { bbox[3]-bbox[2] }
      depth  { bbox[5]-bbox[4] }
    }}
  }}
  # global coordinates and normals
  Coordinate3 {{
    point [ # Dummy normals are needed here -> will be overwritten in the vertex shader
            # If you set coordindates here, these will be used for coin bbox calculation and picking, ... (but will be wrong)
      { " ".join(map(lambda x: str(x), vertexAndNormal.reshape((vertexAndNormal.shape[0]*vertexAndNormal.shape[1],)))) }
    ]
  }}
  Normal {{
    vector [ # Dummy normals are needed here -> will be overwritten in the vertex shader
            # If you set coordindates here, these will be used for coin bbox calculation and picking, ... (but will be wrong)
      { " ".join(map(lambda x: str(x), vertexAndNormal.reshape((vertexAndNormal.shape[0]*vertexAndNormal.shape[1],)))) }
    ]
  }}
  Material {{
    diffuseColor 1 0 0
    specularColor 0.7 0 0
    shininess 0.9
    transparency 0
  }}
  Separator {{
    renderCulling OFF
    ShaderProgram {{
      shaderObject [
        VertexShader {{
          sourceType GLSL_PROGRAM
          sourceProgram "
            #version 130

            out vec3 worldNormal;
            attribute float vertexID_float;
            uniform float data[{6*Nsp+1}];

            mat3 cardan2Rotation(vec3 angle) {{
              float sina = sin(angle.x);
              float cosa = cos(angle.x);
              float sinb = sin(angle.y);
              float cosb = cos(angle.y);
              float sing = sin(angle.z);
              float cosg = cos(angle.z);
              mat3 T;
              T[0][0] = cosb*cosg;
              T[0][1] = -cosb*sing;
              T[0][2] = sinb;
              T[1][0] = cosa*sing+sina*sinb*cosg;
              T[1][1] = cosa*cosg-sina*sinb*sing;
              T[1][2] = -sina*cosb;
              T[2][0] = sina*sing-cosa*sinb*cosg;
              T[2][1] = cosa*sinb*sing+sina*cosg;
              T[2][2] = cosa*cosb;
              return T;
            }}

            vec3 calcPoint(int vertexID) {{
              const vec3 nsp[{Ncs}] = vec3[](
{nspStr}
              );

              int csIdx = int(vertexID/2 % {Ncs});
              int spIdx = int(vertexID/2 / {Ncs});
              vec3 r = vec3(data[spIdx*6+1],data[spIdx*6+2],data[spIdx*6+3]);
              vec3 angle = vec3(data[spIdx*6+4],data[spIdx*6+5],data[spIdx*6+6]);
              mat3 T = transpose(cardan2Rotation(angle));
              vec3 T_nsp = T * nsp[csIdx];
              return r + T_nsp;
            }}

            vec3 calcNormal(int vertexID) {{
              const vec3 normal[{2*Ncs}] = vec3[](
{normalStr}
              );
              const int border[{Ncs}] = int[](
{borderStr}
              );

              int nIdx = int(vertexID % {2*Ncs});
              int spIdx = int(vertexID/2 / {Ncs});
              vec3 angle = vec3(data[spIdx*6+4],data[spIdx*6+5],data[spIdx*6+6]);
              mat3 T = transpose(cardan2Rotation(angle));
              vec3 n = T * normal[nIdx];
              return n;
            }}
             
            void main(void)
            {{
              int vertexID = int(vertexID_float+0.5);

              vec3 c = calcPoint(vertexID);
              vec3 n = calcNormal(vertexID);

              worldNormal = normalize(gl_NormalMatrix * n);
              gl_Position = gl_ModelViewProjectionMatrix * vec4(c, gl_Vertex.w);
              gl_FrontColor = gl_Color;
            }}
          "
          parameter [
            ShaderParameterArray1f {{
              name "data"
              value = USE openmbv_dynamicivbody_data.value
            }}
          ]
        }}
        FragmentShader {{
          sourceType GLSL_PROGRAM
          sourceProgram "
            #version 130

            in vec3 worldNormal;
             
            void directionalLight(in int i,
                                  in vec3 worldNormal,
                                  inout vec4 ambient,
                                  inout vec4 diffuse,
                                  inout vec4 specular)
            {{
              float nDotVP; // normal . light direction
              float nDotHV; // normal . light half vector
              float pf;     // power factor
             
              nDotVP = max(0.0, dot(worldNormal, normalize(vec3(gl_LightSource[i].position))));
             
              if (nDotVP == 0.0)
                pf = 0.0;
              else {{
                nDotHV = max(0.0, dot(worldNormal, vec3(gl_LightSource[i].halfVector)));
                pf = pow(nDotHV, gl_FrontMaterial.shininess);
              }}
             
              ambient += gl_LightSource[i].ambient;
              diffuse += gl_LightSource[i].diffuse * nDotVP;
              specular += gl_LightSource[i].specular * pf;
            }}
             
            void main(void)
            {{
              vec4 ambient = vec4(0.0);
              vec4 diffuse = vec4(0.0);
              vec4 specular = vec4(0.0);
              vec3 color;
             
              directionalLight(0, normalize(worldNormal), ambient, diffuse, specular);//mfmf use more lights and also spot-lights,...
             
              color =
                gl_FrontLightModelProduct.sceneColor.rgb +
                ambient.rgb * gl_FrontMaterial.ambient.rgb +
                diffuse.rgb * gl_Color.rgb +
                specular.rgb * gl_FrontMaterial.specular.rgb;
             
              gl_FragColor = vec4(color, gl_Color.a);
            }}
          "
        }}
      ]
    }}
    VertexAttribute {{
      typeName "SoMFFloat" # Coin only supports short as int type, which is too small.
                           # Coin will convert a int type to float since glVertexAttribPointerARB not glVertexAttriblPointerARB is used
                           # -> use a (none negative) float as ID and convert it to in in the vertex shader
      name "vertexID_float"
      values [ { " ".join(map(lambda x: str(x), range(0,vertexAndNormal.shape[0]))) } ]
    }}
    # tube
    Separator {{
      renderCulling OFF
      renderCaching ON
      IndexedFaceSet {{ # Coin does not pass VertexAttribut's to IndexedTriangleStripSet -> use IndexedFaceSet
        coordIndex [
          { " ".join(map(lambda x: str(x), meshCoordIndex)) }
        ]
        # we cannot use normalIndex/... since the vertex shader calculates exactly one normal per vertex (per Coordinate3)
      }}
    }}
    # outline
    Switch {{
      whichChild = USE openmbv_body_outline_switch.whichChild
      Separator {{
        USE openmbv_body_outline_style
        IndexedLineSet {{
          coordIndex [
            { " ".join(map(lambda x: str(x), tubeCoordIndex)) }
          ]
        }}
      }}
    }}
  }}
  # end-cap
  Separator {{
    NormalBinding {{
      value OVERALL
    }}
    DEF endCapCoords Coordinate3 {{
      point [
        { " ".join(map(lambda x: str(x), nsp.reshape((nsp.shape[0]*nsp.shape[1],)))) }
      ]
    }}
    Separator {{
      Normal {{
        vector 0 1 0
      }}
      ShapeHints {{
        vertexOrdering CLOCKWISE
        shapeType SOLID
      }}
      Transform {{
        translation = DecomposeArray1fToVec3fEngine {{
          startIndex 1
          input = USE openmbv_dynamicivbody_data.value
        }}.output
        rotation = CardanRotationEngine {{
          angle = DecomposeArray1fToVec3fEngine {{
            startIndex 4
            input = USE openmbv_dynamicivbody_data.value
          }}.output
          inverse TRUE
        }}.rotation
      }}
      DEF endCapIndexedTesselationFace IndexedTesselationFace {{
        windingRule ODD
        coordinate = USE endCapCoords.point
        coordIndex [
          { " ".join(map(lambda x: str(x), range(0,Ncs))) }
        ]
      }}
      # outline
      Switch {{
        whichChild = USE openmbv_body_outline_switch.whichChild
        USE openmbv_body_outline_style
        DEF endCapOutLine IndexedLineSet {{
          coordIndex [
            { " ".join(map(lambda x: str(x), range(0,Ncs))) } 0
          ]
        }}
      }}
    }}
    Separator {{
      Normal {{
        vector 0 -1 0
      }}
      ShapeHints {{
        vertexOrdering COUNTERCLOCKWISE
        shapeType SOLID
      }}
      Transform {{
        translation = DecomposeArray1fToVec3fEngine {{
          startIndex {6*(Nsp-1)+1}
          input = USE openmbv_dynamicivbody_data.value
        }}.output
        rotation = CardanRotationEngine {{
          angle = DecomposeArray1fToVec3fEngine {{
            startIndex {6*(Nsp-1)+4}
            input = USE openmbv_dynamicivbody_data.value
          }}.output
          inverse TRUE
        }}.rotation
      }}
      USE endCapIndexedTesselationFace
      # outline
      Switch {{
        whichChild = USE openmbv_body_outline_switch.whichChild
        USE openmbv_body_outline_style
        USE endCapOutLine
      }}
    }}
  }}
}}
'''
print(content)
