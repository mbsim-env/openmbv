#include "group.h"
#include "objectfactory.h"

Group::Group(TiXmlElement* element, H5::Group *h5Parent) : Object(element, h5Parent) {
  setIcon(0, QIcon(":/group.svg"));

  // if xml:base attribute exist => new sub file
  if(element->Attribute("xml:base")) {
    setIcon(0, QIcon(":/h5file.svg"));
    setText(0, element->Attribute("xml:base"));
  }
  // read XML
  TiXmlElement *e=element->FirstChildElement();
  while(e!=0) {
    Object *object=ObjectFactory(e, h5Group);
    addChild(object);
    soSep->addChild(object->getSoSwitch());
    e=e->NextSiblingElement();
  }
}
