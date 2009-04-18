#include "config.h"
#include "group.h"
#include "objectfactory.h"

Group::Group(TiXmlElement* element, H5::Group *h5Parent) : Object(element, h5Parent) {
  iconFile=":/group.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // if xml:base attribute exist => new sub file
  if(element->Attribute("xml:base")) {
    iconFile=":/h5file.svg";
    setIcon(0, QIcon(iconFile.c_str()));
    setText(0, element->Attribute("xml:base"));
  }
  // read XML
  TiXmlElement *e=element->FirstChildElement();
  while(e!=0) {
    Object *object=ObjectFactory(e, h5Group);
    if(object) {
      addChild(object);
      soSep->addChild(object->getSoSwitch());
    }
    e=e->NextSiblingElement();
  }
}

QString Group::getInfo() {
  return Object::getInfo()+
         QString("-----<br/>")+
         QString("<b>Number of Children:</b> %1<br/>").arg(childCount());
}
