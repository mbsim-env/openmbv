#include "config.h"
#include "invisiblebody.h"
#include <Inventor/nodes/SoSphere.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <vector>

using namespace std;

InvisibleBody::InvisibleBody(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/invisiblebody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML

  // create so

  // outline
  soSep->addChild(soOutLineSwitch);
}
