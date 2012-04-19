/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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
#include <QtGui/QMenu>
#include "mainwindow.h"
#include "compoundrigidbody.h"
#include <GL/gl.h>
#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/SoPrimitiveVertex.h>
#include "utils.h"
#include "openmbvcppinterface/body.h"

using namespace std;

map<SoNode*,Body*> Body::bodyMap;

Body::Body(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Object(obj, parentItem, soParent, ind), shilouetteEdgeFirstCall(true), edgeCalc(NULL) {
  body=(OpenMBV::Body*)obj;
  frameSensor=NULL;
  shilouetteEdgeFrameSensor=NULL;
  shilouetteEdgeOrientationSensor=NULL;
  if(obj->getParent()) { // do nothing for rigidbodies inside a compountrigidbody
    // register callback function on frame change
    frameSensor=new SoFieldSensor(frameSensorCB, this);
    frameSensor->attach(MainWindow::getInstance()->getFrame());
    frameSensor->setPriority(0); // is needed for png export
  }

  // switch for outline
  soOutLineSwitch=new SoSwitch;
  soOutLineSwitch->ref(); // add to scene must be done by derived class
  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0)
    soOutLineSwitch->whichChild.setValue(body->getOutLine()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  else
    soOutLineSwitch->whichChild.connectFrom(&((Body*)parentItem)->soOutLineSwitch->whichChild);
  soOutLineSep=new SoSeparator;
  soOutLineSwitch->addChild(soOutLineSep);
  SoLightModel *lm=new SoLightModel;
  soOutLineSep->addChild(lm);
  lm->model.setValue(SoLightModel::BASE_COLOR);
  soOutLineSep->addChild(MainWindow::getInstance()->getOlseColor());
  soOutLineSep->addChild(MainWindow::getInstance()->getOlseDrawStyle());

  // switch for shilouette edge
  soShilouetteEdgeSwitch=new SoSwitch;
  soSep->addChild(soShilouetteEdgeSwitch);
  soShilouetteEdgeSwitch->whichChild.setValue(body->getShilouetteEdge()?SO_SWITCH_ALL:SO_SWITCH_NONE);
  soShilouetteEdgeSep=new SoSeparator;
  soShilouetteEdgeSwitch->addChild(soShilouetteEdgeSep);
  SoLightModel *lm2=new SoLightModel;
  soShilouetteEdgeSep->addChild(lm2);
  lm2->model.setValue(SoLightModel::BASE_COLOR);
  soShilouetteEdgeSep->addChild(MainWindow::getInstance()->getOlseColor());
  soShilouetteEdgeSep->addChild(MainWindow::getInstance()->getOlseDrawStyle());
  soShilouetteEdgeCoord=new SoCoordinate3;
  soShilouetteEdgeSep->addChild(soShilouetteEdgeCoord);
  soShilouetteEdge=new SoIndexedLineSet;
  soShilouetteEdgeSep->addChild(soShilouetteEdge);

  if(dynamic_cast<CompoundRigidBody*>(parentItem)==0) {
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
  
    // GUI
    // register callback function for shilouette edges
    shilouetteEdgeFrameSensor=new SoFieldSensor(shilouetteEdgeFrameOrCameraSensorCB, this);
    shilouetteEdgeOrientationSensor=new SoFieldSensor(shilouetteEdgeFrameOrCameraSensorCB, this);
    if(body->getShilouetteEdge()) {
      soShilouetteEdgeSwitch->whichChild.setValue(SO_SWITCH_ALL);
      shilouetteEdgeFrameSensor->attach(MainWindow::getInstance()->getFrame());
      shilouetteEdgeOrientationSensor->attach(&MainWindow::getInstance()->glViewer->getCamera()->orientation);
      MainWindow::getInstance()->glViewer->getCamera()->orientation.touch();
    }
    else {
      soShilouetteEdgeSwitch->whichChild.setValue(SO_SWITCH_NONE);
      shilouetteEdgeFrameSensor->detach();
      shilouetteEdgeOrientationSensor->detach();
    }

#if 0 
    // GUI editors
    if(!clone) {
      BoolEditor *outLineEditor=new BoolEditor(properties, Utils::QIconCached(":/outline.svg"), "Draw out-line");
      outLineEditor->setOpenMBVParameter(body, &OpenMBV::Body::getOutLine, &OpenMBV::Body::setOutLine);

      BoolEditor *shilouetteEdgeEditor=new BoolEditor(properties, Utils::QIconCached(":/shilouetteedge.svg"), "Draw shilouette edge");
      shilouetteEdgeEditor->setOpenMBVParameter(body, &OpenMBV::Body::getShilouetteEdge, &OpenMBV::Body::setShilouetteEdge);

      ComboBoxEditor *drawMethodEditor=new ComboBoxEditor(properties, Utils::QIconCached(":/lines.svg"), "Draw style",
        boost::assign::tuple_list_of(OpenMBV::Body::filled, "Filled", Utils::QIconCached(":/filled.svg"))
                                    (OpenMBV::Body::lines,  "Lines",  Utils::QIconCached(":/lines.svg"))
                                    (OpenMBV::Body::points, "Points", Utils::QIconCached(":/points.svg"))
      );
      drawMethodEditor->setOpenMBVParameter(body, &OpenMBV::Body::getDrawMethod, &OpenMBV::Body::setDrawMethod);
    }
#endif

    // MFMF hdf5link
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
  for(map<SoNode*,Body*>::iterator it=bodyMap.begin(); it!=bodyMap.end(); it++)
    if(it->second==this) {
      bodyMap.erase(it);
      break;
    }

  // the last Body should reset timeSlider maximum to 0
  if(bodyMap.size()==0)
    MainWindow::getInstance()->timeSlider->setMaximum(0);
}

void Body::frameSensorCB(void *data, SoSensor*) {
  Body* me=(Body*)data;
  static double time=0;
  double newTime=time;
  if(me->drawThisPath) 
    newTime=me->update();
  if(newTime!=time) { // only on first time change
    time=newTime;
    MainWindow::getInstance()->setTime(time);
  }
}

// number of rows / dt
void Body::resetAnimRange(int numOfRows, double dt) {
  if(numOfRows>0) {
    bool existFiles=MainWindow::getInstance()->getTimeSlider()->maximum()>0;
    if(numOfRows-1<MainWindow::getInstance()->getTimeSlider()->maximum() || !existFiles) {
      MainWindow::getInstance()->getTimeSlider()->setMaximum(numOfRows-1);
      if(existFiles) {
        QString str("WARNING! Resetting maximal frame number!");
        MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
        cout<<str.toStdString()<<endl;
      }
    }
    if(MainWindow::getInstance()->getDeltaTime()!=dt || !existFiles) {
      MainWindow::getInstance()->getDeltaTime()=dt;
      if(existFiles) {
        QString str("WARNING! dt in HDF5 datas are not the same!");
        MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
        cout<<str.toStdString()<<endl;
      }
    }
  }
}

void Body::shilouetteEdgeFrameOrCameraSensorCB(void *data, SoSensor* sensor) {
  Body *me=(Body*)data;
  bool preproces=sensor==me->shilouetteEdgeFrameSensor || me->shilouetteEdgeFirstCall==true;
  bool shilouetteCalc=sensor==me->shilouetteEdgeFrameSensor || sensor==me->shilouetteEdgeOrientationSensor || me->shilouetteEdgeFirstCall==true;
  me->shilouetteEdgeFirstCall=false;

  SoCoordinate3 *soEdgeCoordOld=NULL;
  SoIndexedLineSet *soShilouetteEdgeOld=NULL;
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
      me->edgeCalc->preproces(false); // preproces
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
