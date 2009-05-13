#include "openmbvcppinterface/objectfactory.h"
#include "openmbvcppinterface/cuboid.h"
#include "openmbvcppinterface/coilspring.h"

using namespace std;

namespace OpenMBV {

  Object* ObjectFactory::createObject(TiXmlElement *element) {
    if(element==0) return 0;
    if(element->ValueStr()==OPENMBVNS"Cuboid")
      return new Cuboid();
    if(element->ValueStr()==OPENMBVNS"CoilSpring")
      return new CoilSpring();
    return 0;
  }
  
}
