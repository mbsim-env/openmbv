#include "config.h"
#include "body.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <QtGui/QMenu>
#include "mainwindow.h"
#include <GL/gl.h>

using namespace std;

SoSFUInt32 *Body::frame;
bool Body::existFiles=false;

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
  outLine=new QAction(QIcon(":/outline.svg"),"Draw Out-Line", 0);
  outLine->setCheckable(true);
  outLine->setChecked(true);
  connect(outLine,SIGNAL(changed()),this,SLOT(outLineSlot()));
  // draw method action
  drawMethod=new QActionGroup(this);
  drawMethodPolygon=new QAction(QIcon(":/filled.svg"),"Draw Style: Filled", drawMethod);
  drawMethodLine=new QAction(QIcon(":/lines.svg"),"Draw Style: Lines", drawMethod);
  drawMethodPoint=new QAction(QIcon(":/points.svg"),"Draw Style: Points", drawMethod);
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
  menu->addSeparator();
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

// number of rows / dt
void Body::resetAnimRange(int numOfRows, double dt) {
  if(numOfRows-1<MainWindow::timeSlider->maximum() || !existFiles) {
    MainWindow::timeSlider->setMaximum(numOfRows-1);
    if(existFiles)
      cout<<"WARNING! Resetting maximal frame number!"<<endl;
  }
  if(MainWindow::deltaTime!=dt || !existFiles) {
    MainWindow::deltaTime=dt;
    if(existFiles)
      cout<<"WARNING! dt in HDF5 datas are not the same!"<<endl;
  }
  existFiles=true;
}

// convenience: convert e.g. "[3;7;7.9]" to std::vector<double>(3,7,7.9)
vector<double> Body::toVector(string str) {
  for(int i=0; i<str.length(); i++)
    if(str[i]=='[' || str[i]==']' || str[i]==';') str[i]=' ';
  stringstream stream(str);
  double d;
  vector<double> ret;
  while(1) {
    stream>>d;
    if(stream.fail()) break;
    ret.push_back(d);
  }
  return ret;
}

// convenience: create frame so
SoSeparator* Body::soFrame(double size, double offset) {
  SoSeparator *sep=new SoSeparator;
  sep->ref();

  SoBaseColor *col;
  SoLineSet *line;

  // coordinates
  SoCoordinate3 *coord=new SoCoordinate3;
  sep->addChild(coord);
  coord->point.set1Value(0, -size/2+offset*size/2, 0, 0);
  coord->point.set1Value(1, +size/2+offset*size/2, 0, 0);
  coord->point.set1Value(2, 0, -size/2+offset*size/2, 0);
  coord->point.set1Value(3, 0, +size/2+offset*size/2, 0);
  coord->point.set1Value(4, 0, 0, -size/2+offset*size/2);
  coord->point.set1Value(5, 0, 0, +size/2+offset*size/2);

  // x-axis
  col=new SoBaseColor;
  col->rgb=SbColor(1, 0, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(0);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // y-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 1, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(2);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // z-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 0, 1);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(4);
  line->numVertices.setValue(2);
  sep->addChild(line);

  return sep;
}







GLenum tessType;
int tessNumVertices;
SoTriangleStripSet *tessTriangleStrip;
SoIndexedFaceSet *tessTriangleFan;
SoCoordinate3 *tessCoord;

void tessBeginCB(GLenum type, void *data) {
  SoGroup *parent=(SoGroup*)data;
  tessType=type;
  tessNumVertices=0;
  tessCoord=new SoCoordinate3;
  parent->addChild(tessCoord);
  if(tessType==GL_TRIANGLES || tessType==GL_TRIANGLE_STRIP) {
    tessTriangleStrip=new SoTriangleStripSet;
    parent->addChild(tessTriangleStrip);
  }
  if(tessType==GL_TRIANGLE_FAN) {
    tessTriangleFan=new SoIndexedFaceSet;
    parent->addChild(tessTriangleFan);
  }
}

void tessVertexCB(GLdouble *vertex) {
  tessCoord->point.set1Value(tessNumVertices++, vertex[0], vertex[1], vertex[2]);
}

void tessEndCB(void) {
  if(tessType==GL_TRIANGLES)
    for(int i=0; i<tessNumVertices/3; i++)
      tessTriangleStrip->numVertices.set1Value(i, 3);
  if(tessType==GL_TRIANGLE_STRIP)
    tessTriangleStrip->numVertices.set1Value(0, tessNumVertices);
  if(tessType==GL_TRIANGLE_FAN) {
    int j=0;
    for(int i=0; i<tessNumVertices-2; i++) {
      tessTriangleFan->coordIndex.set1Value(j++, 0);
      tessTriangleFan->coordIndex.set1Value(j++, i+1);
      tessTriangleFan->coordIndex.set1Value(j++, i+2);
      tessTriangleFan->coordIndex.set1Value(j++, -1);
    }
  }
}
