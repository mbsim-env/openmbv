#include "frame.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>

Frame::Frame(TiXmlElement *element) : RigidBody(element) {
  setIcon(0, QIcon("frame.svg"));

  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"size");
  double size=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double offset=toVector(e->GetText())[0];

  // create so
  soSep->addChild(soFrame(size, offset));
}
