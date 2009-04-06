#include "group.h"
#include "objectfactory.h"

Group::Group(TiXmlElement* element) : Object(element) {
  setIcon(0, QIcon("group.svg"));

  // read XML
  TiXmlElement *e=element->FirstChildElement();
  while(e!=0) {
    Object *object=ObjectFactory(e);
    addChild(object);
    soSep->addChild(object->getSoSwitch());
    e=e->NextSiblingElement();
  }
}
