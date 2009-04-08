#include "cuboid.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>

using namespace std;

Cuboid::Cuboid(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/cuboid.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"length");
  vector<double> length=toVector(e->GetText());

  // create so
  SoCube *cuboid=new SoCube;
  cuboid->width.setValue(length[0]);
  cuboid->height.setValue(length[1]);
  cuboid->depth.setValue(length[2]);
  soSep->addChild(cuboid);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cuboid);
}
