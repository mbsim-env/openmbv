#include "config.h"
#include "compoundrigidbody.h"
#include "objectfactory.h"

using namespace std;

CompoundRigidBody::CompoundRigidBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/compoundrigidbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  e=e->NextSiblingElement(); // first rigidbody
  while(e!=0) {
    Object *object=ObjectFactory(e, 0, this, soSep);
    e=e->NextSiblingElement(); // next rigidbody
  }

  // create so

  // outline
}
