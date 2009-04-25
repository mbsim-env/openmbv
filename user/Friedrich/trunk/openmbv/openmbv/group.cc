#include "config.h"
#include "group.h"
#include "objectfactory.h"
#include "mainwindow.h"

Group::Group(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Object(element, h5Parent, parentItem, soParent) {
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
    Object *object=ObjectFactory(e, h5Group, this, soSep);
    e=e->NextSiblingElement();
  }
}

QString Group::getInfo() {
  return Object::getInfo()+
         QString("-----<br/>")+
         QString("<b>Number of Children:</b> %1<br/>").arg(childCount());
}
