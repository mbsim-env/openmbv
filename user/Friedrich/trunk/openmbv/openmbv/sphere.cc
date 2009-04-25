#include "config.h"
#include "sphere.h"
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>

using namespace std;

Sphere::Sphere(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/sphere.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"radius");
  double radius=toVector(e->GetText())[0];

  // create so
  SoSphere *sphere=new SoSphere;
  sphere->radius.setValue(radius);
  soSep->addChild(sphere);
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(2*radius,2*radius,2*radius);
  localFrameScale->scaleFactor.setValue(2*radius,2*radius,2*radius);

  // outline
  soSep->addChild(soOutLineSwitch);
}
