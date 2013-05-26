#ifndef _OPENMBV_OBJECTFACTORY_H_
#define _OPENMBV_OBJECTFACTORY_H_

#include <mbxmlutilstinyxml/tinyxml.h>
#include "openmbvcppinterface/object.h"

namespace OpenMBV {

class ObjectFactory {
  public:
    static Object* createObject(MBXMLUtils::TiXmlElement *element);
};

}

#endif
