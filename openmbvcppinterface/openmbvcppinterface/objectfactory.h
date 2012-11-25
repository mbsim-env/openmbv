#ifndef _OPENMBV_OBJECTFACTORY_H_
#define _OPENMBV_OBJECTFACTORY_H_

#include <mbxmlutilstinyxml/tinyxml.h>
#include "openmbvcppinterface/object.h"

namespace OpenMBV {

class ObjectFactory {
  protected:
    typedef std::pair<std::string, std::string> P_NSPRE;
    typedef std::pair<double, P_NSPRE> P_PRINSPRE;
    typedef std::multimap<double, P_NSPRE> MM_PRINSPRE;
  public:
    static Object* createObject(TiXmlElement *element);
    static MM_PRINSPRE& getPriorityNamespacePrefix();
};

}

#endif
