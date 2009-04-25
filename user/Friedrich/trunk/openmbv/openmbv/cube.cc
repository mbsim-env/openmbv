#include "config.h"
#include "cube.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>

using namespace std;

Cube::Cube(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/cube.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"length");
  double length=toVector(e->GetText())[0];

  // create so
  SoCube *cube=new SoCube;
  cube->width.setValue(length);
  cube->height.setValue(length);
  cube->depth.setValue(length);
  soSep->addChild(cube);
  // scale ref/localFrame
  refFrameScale->scaleFactor.setValue(length,length,length);
  localFrameScale->scaleFactor.setValue(length,length,length);

  // outline
  soSep->addChild(soOutLineSwitch);
  soOutLineSep->addChild(cube);
}
