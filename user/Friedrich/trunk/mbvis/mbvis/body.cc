#include "body.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <QtGui/QMenu>

using namespace std;

SoSFUInt32 *Body::frame;

Body::Body(TiXmlElement *element, H5::Group *h5Parent) : Object(element, h5Parent) {
  // read XML
  TiXmlElement *e=element->FirstChildElement(MBVISNS"hdf5Link");
  if(e); // hdf6Link

  // register callback function on frame change
  SoFieldSensor *sensor=new SoFieldSensor(frameSensorCB, this);
  sensor->attach(frame);

  // switch for outline
  soOutLineSwitch=new SoSwitch;
  soOutLineSwitch->ref(); // add to scene must be done by derived class
  soOutLineSwitch->whichChild.setValue(SO_SWITCH_ALL);
  soOutLineSep=new SoSeparator;
  soOutLineSwitch->addChild(soOutLineSep);
  SoBaseColor *color=new SoBaseColor;
  color->rgb.setValue(0,0,0);
  soOutLineSep->addChild(color);
  SoDrawStyle *style=new SoDrawStyle;
  style->style.setValue(SoDrawStyle::LINES);
  soOutLineSep->addChild(style);

  // GUI
  outLine=new QAction("Draw Out-Line", 0);
  outLine->setCheckable(true);
  outLine->setChecked(true);
  connect(outLine,SIGNAL(changed()),this,SLOT(outLineSlot()));
}

void Body::frameSensorCB(void *data, SoSensor*) {
  Body* me=(Body*)data;
  if(me->drawThisPath) 
    me->update();
}

QMenu* Body::createMenu() {
  QMenu* menu=Object::createMenu();
  menu->addSeparator();
  QAction *type=new QAction("Properties from: Body", menu);
  type->setEnabled(false);
  menu->addAction(type);
  menu->addAction(outLine);
  return menu;
}

void Body::outLineSlot() {
  if(outLine->isChecked())
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_NONE);
}
