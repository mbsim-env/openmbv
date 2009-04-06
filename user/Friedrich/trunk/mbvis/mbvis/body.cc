#include "body.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoScale.h>
#include <QtGui/QMenu>

using namespace std;

Body::Body(TiXmlElement *element) : Object(element) {
  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"hdf5Link");
  if(e); // hdf6Link

  // register callback function on frame change
  SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
  sensor->attach(frame);
}

void Body::frameSensorCB(void *data, SoSensor*) {
  Body* me=(Body*)data;
  if(me->drawThisPath) 
    me->update();
}
