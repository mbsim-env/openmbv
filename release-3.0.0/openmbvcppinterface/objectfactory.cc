#include "openmbvcppinterface/objectfactory.h"
#include "openmbvcppinterface/cuboid.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/frustum.h"
#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/compoundrigidbody.h"
#include "openmbvcppinterface/cube.h"
#include "openmbvcppinterface/extrusion.h"
#include "openmbvcppinterface/frame.h"
#include "openmbvcppinterface/invisiblebody.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/objbody.h"
#include "openmbvcppinterface/path.h"
#include "openmbvcppinterface/rotation.h"
#include "openmbvcppinterface/sphere.h"
#include "openmbvcppinterface/spineextrusion.h"

using namespace std;

namespace OpenMBV {

  Object* ObjectFactory::createObject(TiXmlElement *element) {
    if(element==0) return 0;
    if(element->ValueStr()==OPENMBVNS"Cuboid") return new Cuboid;
    if(element->ValueStr()==OPENMBVNS"CoilSpring") return new CoilSpring;
    if(element->ValueStr()==OPENMBVNS"Frustum") return new Frustum;
    if(element->ValueStr()==OPENMBVNS"Arrow") return new Arrow;
    if(element->ValueStr()==OPENMBVNS"CompoundRigidBody") return new CompoundRigidBody;
    if(element->ValueStr()==OPENMBVNS"Cube") return new Cube;
    if(element->ValueStr()==OPENMBVNS"Extrusion") return new Extrusion;
    if(element->ValueStr()==OPENMBVNS"Frame") return new Frame;
    if(element->ValueStr()==OPENMBVNS"InvisibleBody") return new InvisibleBody;
    if(element->ValueStr()==OPENMBVNS"IvBody") return new IvBody;
    if(element->ValueStr()==OPENMBVNS"ObjBody") return new ObjBody;
    if(element->ValueStr()==OPENMBVNS"Path") return new Path;
    if(element->ValueStr()==OPENMBVNS"Rotation") return new Rotation;
    if(element->ValueStr()==OPENMBVNS"Sphere") return new Sphere;
    if(element->ValueStr()==OPENMBVNS"SpineExtrusion") return new SpineExtrusion;
    return 0;
  }
  
}
