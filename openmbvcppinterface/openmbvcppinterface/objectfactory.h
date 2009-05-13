#ifndef _OPENMBV_OBJECTFACTORY_H_
#define _OPENMBV_OBJECTFACTORY_H_

#include "openmbvcppinterfacetinyxml/tinyxml.h"
#include "openmbvcppinterface/object.h"

namespace OpenMBV {

class ObjectFactory {
  public:
    static Object* createObject(TiXmlElement *element);
};

}

#endif
