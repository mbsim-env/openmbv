#include "cuboid.h"
#include <Inventor/nodes/SoCube.h>
#include <vector>

using namespace std;

Cuboid::Cuboid(TiXmlElement *element) : RigidBody(element) {
  setIcon(0, QIcon("cuboid.svg"));

  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"length");
  vector<double> length=toVector(e->GetText());

  // create so
  SoCube *cuboid=new SoCube;
  cuboid->width=length[0];
  cuboid->height=length[1];
  cuboid->depth=length[2];
  soSep->addChild(cuboid);
}
