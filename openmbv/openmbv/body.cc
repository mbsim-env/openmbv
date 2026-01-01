/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include "config.h"
#include "body.h"
#include "utils.h"
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/actions/SoSearchAction.h>
#include "SoSpecial.h"
#include <QMenu>
#include "mainwindow.h"
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN // GL/gl.h includes windows.h on Windows -> avoid full header -> WIN32_LEAN_AND_MEAN
#  endif
#endif
#include <GL/gl.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/SoPrimitiveVertex.h>
#include "utils.h"
#include "openmbvcppinterface/body.h"
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

map<SoNode*,Body*> Body::bodyMap;

Body::Body(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind), shilouetteEdgeFirstCall(true), edgeCalc(nullptr) {
  body=std::static_pointer_cast<OpenMBV::Body>(obj);
  frameSensor=nullptr;
  shilouetteEdgeFrameSensor=nullptr;
  shilouetteEdgeOrientationSensor=nullptr;
  std::shared_ptr<OpenMBV::Group> p=obj->getParent().lock();
  if(p) { // do nothing for rigidbodies inside a compountrigidbody
    // register callback function on frame change
    frameSensor=new SoFieldSensor(frameSensorCB, this);
    frameSensor->attach(&MainWindow::getInstance()->getFrame());
    frameSensor->setPriority(0); // is needed for png export
  }

  // switch for outline
  soOutLineSwitch=new SoSwitch;
  soOutLineSwitch->ref(); // add to scene must be done by derived class
  soOutLineSwitch->whichChild.setValue(body->getOutLine()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soOutLineSep=new SoSeparator;
  soOutLineSwitch->addChild(soOutLineSep);
  soOutLineStyle = new SoGroup;
  soOutLineSep->addChild(soOutLineStyle);
  auto *lm=new SoLightModel;
  soOutLineStyle->addChild(lm);
  lm->model.setValue(SoLightModel::BASE_COLOR);
  soOutLineStyle->addChild(MainWindow::getInstance()->getOlseColor());
  soOutLineStyle->addChild(MainWindow::getInstance()->getOlseDrawStyle());
  // render outlines without backface culling
  auto *sh=new SoShapeHints;
  soOutLineStyle->addChild(sh);
  sh->shapeType=SoShapeHints::UNKNOWN_SHAPE_TYPE;

  // switch for shilouette edge
  soShilouetteEdgeSwitch=new SoSwitch;
  soSep->addChild(soShilouetteEdgeSwitch);
  soShilouetteEdgeSwitch->whichChild.setValue(body->getShilouetteEdge()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soShilouetteEdgeSep=new SoSeparator;
  soShilouetteEdgeSwitch->addChild(soShilouetteEdgeSep);
  auto *lm2=new SoLightModel;
  soShilouetteEdgeSep->addChild(lm2);
  lm2->model.setValue(SoLightModel::BASE_COLOR);
  soShilouetteEdgeSep->addChild(MainWindow::getInstance()->getOlseColor());
  soShilouetteEdgeSep->addChild(MainWindow::getInstance()->getOlseDrawStyle());
  soShilouetteEdgeCoord=new SoCoordinate3;
  soShilouetteEdgeSep->addChild(soShilouetteEdgeCoord);
  soShilouetteEdge=new SoIndexedLineSet;
  soShilouetteEdgeSep->addChild(soShilouetteEdge);

  // add to map for finding this object by the soSep SoNode
  bodyMap.insert(pair<SoNode*, Body*>(soSep,this));

  // draw method
  drawStyle=new SoDrawStyle;
  soSep->addChild(drawStyle);
  switch(body->getDrawMethod()) {
    case OpenMBV::Body::filled: drawStyle->style.setValue(SoDrawStyle::FILLED); break;
    case OpenMBV::Body::lines: drawStyle->style.setValue(SoDrawStyle::LINES); break;
    case OpenMBV::Body::points: drawStyle->style.setValue(SoDrawStyle::POINTS); break;
  }
  drawStyle->pointSize.setValue(body->getPointSize());
  drawStyle->lineWidth.setValue(body->getLineWidth());
  
  // GUI
  // register callback function for shilouette edges
  shilouetteEdgeFrameSensor=new SoFieldSensor(shilouetteEdgeFrameOrCameraSensorCB, this);
  shilouetteEdgeOrientationSensor=new SoFieldSensor(shilouetteEdgeFrameOrCameraSensorCB, this);
  if(body->getShilouetteEdge()) {
    soShilouetteEdgeSwitch->whichChild.setValue(SO_SWITCH_ALL);
    shilouetteEdgeFrameSensor->attach(&MainWindow::getInstance()->getFrame());
    shilouetteEdgeOrientationSensor->attach(&MainWindow::getInstance()->glViewer->getCamera()->orientation);
    MainWindow::getInstance()->glViewer->getCamera()->orientation.touch();
  }
  else {
    soShilouetteEdgeSwitch->whichChild.setValue(SO_SWITCH_NONE);
    shilouetteEdgeFrameSensor->detach();
    shilouetteEdgeOrientationSensor->detach();
  }
}

Body::~Body() {
  // delete scene graph
  SoSearchAction sa;
  sa.setInterest(SoSearchAction::FIRST);
  sa.setNode(soOutLineSwitch);
  sa.apply(MainWindow::getInstance()->getSceneRoot());
  SoPath *p=sa.getPath();
  if(p) ((SoGroup*)p->getNodeFromTail(1))->removeChild(soOutLineSwitch);
  // delete the rest
  delete edgeCalc;
  delete frameSensor;
  delete shilouetteEdgeFrameSensor;
  delete shilouetteEdgeOrientationSensor;
  soOutLineSwitch->unref();

  // remove from map
  for(auto it=bodyMap.begin(); it!=bodyMap.end(); it++)
    if(it->second==this) {
      bodyMap.erase(it);
      break;
    }

  // the last Body should reset timeSlider maximum to 0
  if(bodyMap.empty())
    MainWindow::getInstance()->timeSlider->setTotalMaximum(0);
}

void Body::createProperties() {
  Object::createProperties();

  // GUI editors
  if(!clone) {
    auto *outLineEditor=new BoolEditor(properties, Utils::QIconCached("outline.svg"), "Draw out-line", "Body::outLine");
    outLineEditor->setOpenMBVParameter(body, &OpenMBV::Body::getOutLine, &OpenMBV::Body::setOutLine);
    properties->addPropertyAction(outLineEditor->getAction());

    auto *shilouetteEdgeEditor=new BoolEditor(properties, Utils::QIconCached("shilouetteedge.svg"), "Draw shilouette edge", "Body::shilouetteEdge");
    shilouetteEdgeEditor->setOpenMBVParameter(body, &OpenMBV::Body::getShilouetteEdge, &OpenMBV::Body::setShilouetteEdge);
    properties->addPropertyAction(shilouetteEdgeEditor->getAction());

    auto *drawMethodEditor=new ComboBoxEditor(properties, Utils::QIconCached("lines.svg"), "Draw style", {
      make_tuple(OpenMBV::Body::filled, "Filled", Utils::QIconCached("filled.svg"), "Body::drawStyle::filled"),
      make_tuple(OpenMBV::Body::lines,  "Lines",  Utils::QIconCached("lines.svg"),  "Body::drawStyle::lines"),
      make_tuple(OpenMBV::Body::points, "Points", Utils::QIconCached("points.svg"), "Body::drawStyle::points")
    });
    drawMethodEditor->setOpenMBVParameter(body, &OpenMBV::Body::getDrawMethod, &OpenMBV::Body::setDrawMethod);
    properties->addPropertyActionGroup(drawMethodEditor->getActionGroup());

    auto *pointSizeEditor=new FloatEditor(properties, Utils::QIconCached("pointsize.svg"), "Define point size");
    pointSizeEditor->setRange(0, DBL_MAX);
    pointSizeEditor->setOpenMBVParameter(body, &OpenMBV::Body::getPointSize, &OpenMBV::Body::setPointSize);

    auto *lineWidthEditor=new FloatEditor(properties, Utils::QIconCached("linewidth.svg"), "Define line width");
    lineWidthEditor->setRange(0, DBL_MAX);
    lineWidthEditor->setOpenMBVParameter(body, &OpenMBV::Body::getLineWidth, &OpenMBV::Body::setLineWidth);
  }
}

void Body::frameSensorCB(void *data, SoSensor*) {
  auto* me=(Body*)data;
  static double time=0;
  double newTime=time;
  if(me->drawThisPath)
    newTime=me->update();
  if(!isnan(newTime) && newTime!=time && !me->object->getEnvironment()) { // only on first time change and for environment body's (which have hdf5 data)
    time=newTime;
    MainWindow::getInstance()->setTime(time);
  }
}

// number of rows / dt
void Body::resetAnimRange(int numOfRows, double dt) {
  if(numOfRows>0) {
    bool existFiles=MainWindow::getInstance()->getTimeSlider()->totalMaximum()>0;
    if(numOfRows-1<MainWindow::getInstance()->getTimeSlider()->totalMaximum() || !existFiles) {
      MainWindow::getInstance()->timeSlider->setTotalMaximum(numOfRows-1);
      MainWindow::getInstance()->frameMinSB->setMaximum(numOfRows-1);
      MainWindow::getInstance()->frameMaxSB->setMaximum(numOfRows-1);
      if(existFiles) {
        QString str("Resetting maximal frame number!");
        MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
        msg(Warn)<<str.toStdString()<<endl;
      }
    }
    if(dt!=0 && (MainWindow::getInstance()->getDeltaTime()!=dt || !existFiles)) {
      MainWindow::getInstance()->getDeltaTime()=dt;
      if(existFiles) {
        QString str("dt in HDF5 datas are not the same!");
        MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
        msg(Warn)<<str.toStdString()<<endl;
      }
    }
  }
}

void Body::shilouetteEdgeFrameOrCameraSensorCB(void *data, SoSensor* sensor) {
  auto *me=(Body*)data;
  bool preproces=sensor==me->shilouetteEdgeFrameSensor || me->shilouetteEdgeFirstCall;
  bool shilouetteCalc=sensor==me->shilouetteEdgeFrameSensor || sensor==me->shilouetteEdgeOrientationSensor || me->shilouetteEdgeFirstCall;
  me->shilouetteEdgeFirstCall=false;

  SoCoordinate3 *soEdgeCoordOld=nullptr;
  SoIndexedLineSet *soShilouetteEdgeOld=nullptr;
  SbVec3f n;
  if(preproces) { // new preprocessing required: do all except preprocessing and coord exchange
    soEdgeCoordOld=me->soShilouetteEdgeCoord;
    int outLineSaved=me->soOutLineSwitch->whichChild.getValue(); // save outline
    me->soOutLineSwitch->whichChild.setValue(SO_SWITCH_NONE); // disable outline
    delete me->edgeCalc;
    me->edgeCalc=new EdgeCalculation(me->soSep, false); // collect edge data
    me->soOutLineSwitch->whichChild.setValue(outLineSaved); // restore outline
  }
  if(shilouetteCalc) { // new shilouette edge required: to all except shilouette edge calculation and line set exchange
    SbRotation r=MainWindow::getInstance()->glViewer->getCamera()->orientation.getValue(); // camera orientation
    r*=((SoSFRotation*)(MainWindow::getInstance()->cameraOrientation->outRotation[0]))->getValue(); // camera orientation relative to "Move Camera with Body"
    r.multVec(SbVec3f(0,0,-1),n); // a vector normal to the viewport in the world frame
    soShilouetteEdgeOld=me->soShilouetteEdge;
  }
  { // THREAD THIS OUT in further development: preproces and edge calculation
    if(preproces)
      me->edgeCalc->preproces(me->object->getFullName(), false); // preproces
    if(shilouetteCalc)
      me->edgeCalc->calcShilouetteEdges(n); // calc shilouette edges for normal n
  }
  { // WAIT FOR PREVIOUS THREAD in further development: exchange coord and line set
    if(preproces)
      me->soShilouetteEdgeSep->replaceChild(soEdgeCoordOld, me->soShilouetteEdgeCoord=me->edgeCalc->getCoordinates()); // replace coords
    if(shilouetteCalc)
      me->soShilouetteEdgeSep->replaceChild(soShilouetteEdgeOld, me->soShilouetteEdge=me->edgeCalc->getShilouetteEdges()); // replace shilouette edges
  }
}

}
