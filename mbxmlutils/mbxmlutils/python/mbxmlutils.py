from math import sin, cos
from numpy import array, zeros, append



def registerPath(path):
  # load the libmbxmlutils-eval-python.so ones
  if registerPath.dll==None:
    import ctypes
    registerPath.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-python.so")
  # call the mbxmlutilsPyEvalRegisterPath function in this lib
  ret=registerPath.dll.mbxmlutilsPyEvalRegisterPath(path)
registerPath.dll=None



def load(filename):
  import csv
  with open(filename, 'r') as fileObj:
    reader=csv.reader(fileObj, delimiter=' ')
    ret=None
    for row in reader:
      if ret is None:
        ret=zeros((0, len(row)))
      ret=append(ret, [list(map(float, row))], 0)
  return ret



def cardan(*argv):
  if len(argv)==3:
    alpha=argv[0]
    beta=argv[1]
    gamma=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    alpha=argv[0][0];
    beta=argv[0][1];
    gamma=argv[0][2];
  else:
    raise RuntimError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return array([[cos(beta)*cos(gamma),
               -cos(beta)*sin(gamma),
               sin(beta)],
               [cos(alpha)*sin(gamma)+sin(alpha)*sin(beta)*cos(gamma),
               cos(alpha)*cos(gamma)-sin(alpha)*sin(beta)*sin(gamma),
               -sin(alpha)*cos(beta)],
               [sin(alpha)*sin(gamma)-cos(alpha)*sin(beta)*cos(gamma),
               cos(alpha)*sin(beta)*sin(gamma)+sin(alpha)*cos(gamma),
               cos(alpha)*cos(beta)]])



def euler(*argv):
  if len(argv)==3:
    PHI=argv[0]
    theta=argv[1]
    phi=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    PHI=argv[0][0]
    theta=argv[0][1]
    phi=argv[0][2]
  else:
    raise RuntimeError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return array([[cos(phi)*cos(PHI)-sin(phi)*cos(theta)*sin(PHI),
               -cos(phi)*cos(theta)*sin(PHI)-sin(phi)*cos(PHI),
               sin(theta)*sin(PHI)],
               [cos(phi)*sin(PHI)+sin(phi)*cos(theta)*cos(PHI),
               cos(phi)*cos(theta)*cos(PHI)-sin(phi)*sin(PHI),
               -sin(theta)*cos(PHI)],
               [sin(phi)*sin(theta),
               cos(phi)*sin(theta),
               cos(theta)]])



def rotateAboutX(phi):
  return array([[1,0,0],
               [0,cos(phi),-sin(phi)],
               [0,sin(phi),cos(phi)]])



def rotateAboutY(phi):
  return array([[cos(phi),0,sin(phi)],
               [0,1,0],
               [-sin(phi),0,cos(phi)]])



def rotateAboutZ(phi):
  return array([[cos(phi),-sin(phi),0],
               [sin(phi),cos(phi),0],
               [0,0,1]])
