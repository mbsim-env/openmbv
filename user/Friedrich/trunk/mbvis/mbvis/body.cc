#include "body.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
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

  // draw method
  drawStyle=new SoDrawStyle;
  soSep->addChild(drawStyle);

  // GUI
  // draw outline action
  outLine=new QAction("Draw Out-Line", 0);
  outLine->setCheckable(true);
  outLine->setChecked(true);
  connect(outLine,SIGNAL(changed()),this,SLOT(outLineSlot()));
  // draw method action
  drawMethod=new QActionGroup(this);
  drawMethodPolygon=new QAction("Draw Style: Filled", drawMethod);
  drawMethodLine=new QAction("Draw Style: Lines", drawMethod);
  drawMethodPoint=new QAction("Draw Style: Points", drawMethod);
  drawMethodPolygon->setCheckable(true);
  drawMethodPolygon->setData(QVariant(filled));
  drawMethodLine->setCheckable(true);
  drawMethodLine->setData(QVariant(lines));
  drawMethodPoint->setCheckable(true);
  drawMethodPoint->setData(QVariant(points));
  drawMethodPolygon->setChecked(true);
  connect(drawMethod,SIGNAL(triggered(QAction*)),this,SLOT(drawMethodSlot(QAction*)));
}

void Body::frameSensorCB(void *data, SoSensor*) {
  Body* me=(Body*)data;
  if(me->drawThisPath) 
    me->update();
}

QMenu* Body::createMenu() {
  QMenu* menu=Object::createMenu();
  menu->addSeparator()->setText("Properties from: Body");
  menu->addAction(outLine);
  menu->addAction(drawMethodPolygon);
  menu->addAction(drawMethodLine);
  menu->addAction(drawMethodPoint);
  return menu;
}

void Body::outLineSlot() {
  if(outLine->isChecked())
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_ALL);
  else
    soOutLineSwitch->whichChild.setValue(SO_SWITCH_NONE);
}

void Body::drawMethodSlot(QAction* action) {
  DrawStyle ds=(DrawStyle)action->data().toInt();
  if(ds==filled)
    drawStyle->style.setValue(SoDrawStyle::FILLED);
  else if(ds==lines)
    drawStyle->style.setValue(SoDrawStyle::LINES);
  else
    drawStyle->style.setValue(SoDrawStyle::POINTS);
}
