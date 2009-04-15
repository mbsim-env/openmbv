#include "config.h"
#include "cube.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>

using namespace std;

Cube::Cube(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/cube.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"length");
  double length=toVector(e->GetText())[0];

  // create so
  SoCube *cube=new SoCube;
  cube->width.setValue(length);
  cube->height.setValue(length);
  cube->depth.setValue(length);
  soSep->addChild(cube);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cube);
}
