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

#include "mytouchwidget.h"
#include "touchwidget_impl.h"
#include "Inventor/nodes/SoCamera.h"
#include "Inventor/nodes/SoOrthographicCamera.h"
#include "Inventor/nodes/SoPerspectiveCamera.h"
#include "Inventor/actions/SoRayPickAction.h"
#include <Inventor/SoPickedPoint.h>
#include "mainwindow.h"
#include "fmatvec/atom.h"
#include "utils.h"
#include <QMenu>
#include <QMetaMethod>
#include <cmath>
#include <iostream>

#define DEBUG(x)

using namespace std;

namespace OpenMBVGUI {

MyTouchWidget::MyTouchWidget(QWidget *parent) : TouchWidget<QWidget>(parent) {
  setAttribute(Qt::WA_Hover, true);
  setCursor(Qt::BlankCursor);
  mouseLeftMoveAction=static_cast<MouseMoveAction>(appSettings->get<int>(AppSettings::mouseLeftMoveAction));
  mouseRightMoveAction=static_cast<MouseMoveAction>(appSettings->get<int>(AppSettings::mouseRightMoveAction));
  mouseMidMoveAction=static_cast<MouseMoveAction>(appSettings->get<int>(AppSettings::mouseMidMoveAction));
  mouseLeftClickAction=static_cast<MouseClickAction>(appSettings->get<int>(AppSettings::mouseLeftClickAction));
  mouseRightClickAction=static_cast<MouseClickAction>(appSettings->get<int>(AppSettings::mouseRightClickAction));
  mouseMidClickAction=static_cast<MouseClickAction>(appSettings->get<int>(AppSettings::mouseMidClickAction));
  touchTapAction=static_cast<TouchTapAction>(appSettings->get<int>(AppSettings::touchTapAction));
  touchLongTapAction=static_cast<TouchTapAction>(appSettings->get<int>(AppSettings::touchLongTapAction));
  touchMove1Action=static_cast<TouchMoveAction>(appSettings->get<int>(AppSettings::touchMove1Action));
  touchMove2Action=static_cast<TouchMoveAction>(appSettings->get<int>(AppSettings::touchMove2Action));
}

void MyTouchWidget::updateCursorPos(const QPoint &mousePos) {
  int x=mousePos.x();
  int y=mousePos.y();
  auto size=MainWindow::getInstance()->glViewer->getViewportRegion().getViewportSizePixels();
  if(MainWindow::getInstance()->dialogStereo) {
    if(MainWindow::getInstance()->glViewerWG==this)
      x=x/2;
    else
      x=(size[0]+x)/2;
  }
  SbVec2f pos(static_cast<double>(x)/size[0],1.0-static_cast<double>(y)/size[1]);
  if(!MainWindow::getInstance()->dialogStereo) {
    int largeIdx=size[1]>size[0] ? 1 : 0;
    int smallIdx=size[1]>size[0] ? 0 : 1;
    auto fac=static_cast<float>(size[largeIdx])/size[smallIdx];
    pos[largeIdx]=pos[largeIdx]*fac-(fac-1.0)/2.0;
  }
  auto vv=MainWindow::getInstance()->glViewer->getCamera()->getViewVolume();
  SbVec3f near, far;
  vv.projectPointToLine(SbVec2f(pos[0],pos[1]), near, far);
  SbVec3f cursorPos;
  if(MainWindow::getInstance()->glViewer->getCamera()->getTypeId()==SoOrthographicCamera::getClassTypeId())
    cursorPos=near*(1-0.5)+far*0.5;
  else
    cursorPos=near*(1-relCursorZ)+far*relCursorZ;
  MainWindow::getInstance()->setCursorPos(&cursorPos);
}

bool MyTouchWidget::event(QEvent *event) {
  switch(event->type()) {
    case QEvent::HoverEnter:
    case QEvent::HoverMove: {
      auto mouseEvent=static_cast<QHoverEvent*>(event);
      updateCursorPos(mouseEvent->pos());
      break;
    }
    case QEvent::HoverLeave:
      MainWindow::getInstance()->setCursorPos();
      break;
    default: break;
  }
  return TouchWidget<QWidget>::event(event);
}



void MyTouchWidget::mouseLeftClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse leftclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  bool altDown=modifiers & Qt::AltModifier;
  switch(mouseLeftClickAction) {
    case MouseClickAction::Select:
      selectObject(pos, ctrlDown, altDown);
      break;
    case MouseClickAction::Context: 
      // if Ctrl down, show context menu (for all selected objects)
      if(ctrlDown)
        MainWindow::getInstance()->execPropertyMenu();
      else
        selectObjectAndShowContextMenu(pos, altDown);
      break;
    case MouseClickAction::SeekToPoint:
      seekToPoint(pos);
      break;
  }
}

void MyTouchWidget::mouseRightClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse rightclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  bool altDown=modifiers & Qt::AltModifier;
  switch(mouseRightClickAction) {
    case MouseClickAction::Select:
      selectObject(pos, ctrlDown, altDown);
      break;
    case MouseClickAction::Context: 
      // if Ctrl down, show context menu (for all selected objects)
      if(ctrlDown)
        MainWindow::getInstance()->execPropertyMenu();
      else
        selectObjectAndShowContextMenu(pos, altDown);
      break;
    case MouseClickAction::SeekToPoint:
      seekToPoint(pos);
      break;
  }
}

void MyTouchWidget::mouseMidClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse midclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  bool altDown=modifiers & Qt::AltModifier;
  switch(mouseMidClickAction) {
    case MouseClickAction::Select:
      selectObject(pos, ctrlDown, altDown);
      break;
    case MouseClickAction::Context: 
      // if Ctrl down, show context menu (for all selected objects)
      if(ctrlDown)
        MainWindow::getInstance()->execPropertyMenu();
      else
        selectObjectAndShowContextMenu(pos, altDown);
      break;
    case MouseClickAction::SeekToPoint:
      seekToPoint(pos);
      break;
  }
}

void MyTouchWidget::mouseLeftDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse leftDoubleclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  openPropertyDialog(pos);
}

void MyTouchWidget::mouseRightDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse rightDoubleclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
}

void MyTouchWidget::mouseMidDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse midDoubleclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
}

void MyTouchWidget::mouseLeftMoveSave(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse leftsave"<<endl;)
  switch(mouseLeftMoveAction) {
    case MouseMoveAction::Rotate: rotateInit(); break;
    case MouseMoveAction::Translate: translateInit(); break;
    case MouseMoveAction::Zoom: zoomInit(); break;
  }
}

void MyTouchWidget::mouseRightMoveSave(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse rightsave"<<endl;)
  switch(mouseRightMoveAction) {
    case MouseMoveAction::Rotate: rotateInit(); break;
    case MouseMoveAction::Translate: translateInit(); break;
    case MouseMoveAction::Zoom: zoomInit(); break;
  }
}

void MyTouchWidget::mouseMidMoveSave(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse midsave"<<endl;)
  switch(mouseMidMoveAction) {
    case MouseMoveAction::Rotate: rotateInit(); break;
    case MouseMoveAction::Translate: translateInit(); break;
    case MouseMoveAction::Zoom: zoomInit(); break;
  }
}

void MyTouchWidget::mouseLeftMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse leftReset"<<endl;)
  rotateReset();
  switch(mouseLeftMoveAction) {
    case MouseMoveAction::Rotate: rotateReset(); break;
    case MouseMoveAction::Translate: translateReset(); break;
    case MouseMoveAction::Zoom: zoomReset(); break;
  }
}

void MyTouchWidget::mouseRightMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse rightReset"<<endl;)
  switch(mouseRightMoveAction) {
    case MouseMoveAction::Rotate: rotateReset(); break;
    case MouseMoveAction::Translate: translateReset(); break;
    case MouseMoveAction::Zoom: zoomReset(); break;
  }
}

void MyTouchWidget::mouseMidMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse midReset"<<endl;)
  switch(mouseMidMoveAction) {
    case MouseMoveAction::Rotate: rotateReset(); break;
    case MouseMoveAction::Translate: translateReset(); break;
    case MouseMoveAction::Zoom: zoomReset(); break;
  }
}

void MyTouchWidget::mouseLeftMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse leftMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  switch(mouseLeftMoveAction) {
    case MouseMoveAction::Rotate:
      if(!ctrlDown)
        rotateInScreenAxis(rel);
      else
        rotateInScreenPlane(rel.y()*rotAnglePerPixel*M_PI/180 );
      break;
    case MouseMoveAction::Translate:
      translate(rel);
      break;
    case MouseMoveAction::Zoom:
      if(!ctrlDown)
        zoomCameraAngle(rel.y());
      else
        zoomCameraFocalDist(rel.y());
      break;
  }
}

void MyTouchWidget::mouseRightMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse rightMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  switch(mouseRightMoveAction) {
    case MouseMoveAction::Rotate:
      if(!ctrlDown)
        rotateInScreenAxis(rel);
      else
        rotateInScreenPlane(rel.y()*rotAnglePerPixel*M_PI/180 );
      break;
    case MouseMoveAction::Translate:
      translate(rel);
      break;
    case MouseMoveAction::Zoom:
      if(!ctrlDown)
        zoomCameraAngle(rel.y());
      else
        zoomCameraFocalDist(rel.y());
      break;
  }
}

void MyTouchWidget::mouseMidMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse midMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  switch(mouseMidMoveAction) {
    case MouseMoveAction::Rotate:
      if(!ctrlDown)
        rotateInScreenAxis(rel);
      else
        rotateInScreenPlane(rel.y()*rotAnglePerPixel*M_PI/180 );
      break;
    case MouseMoveAction::Translate:
      translate(rel);
      break;
    case MouseMoveAction::Zoom:
      if(!ctrlDown)
        zoomCameraAngle(rel.y());
      else
        zoomCameraFocalDist(rel.y());
      break;
  }
}

void MyTouchWidget::mouseWheel(Qt::KeyboardModifiers modifiers, double relAngle, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse wheel deltaAngle="<<relAngle<<"°"<<endl;)
  if(modifiers & Qt::ControlModifier) {//mfmf qsetting
    relCursorZ+=relAngle/15/100;//mfmf qsetting
    if(relCursorZ<=1e-6) relCursorZ=1e-6;
    if(relCursorZ>=1-1e-6) relCursorZ=1-1e-6;
    updateCursorPos(pos);
  }
  else {//mfmf qsetting
    int steps=lround(relAngle/15);
    changeFrame(steps);
  }
}



void MyTouchWidget::touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch tap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  bool altDown=modifiers & Qt::AltModifier;
  switch(touchTapAction) {
    case TouchTapAction::Select:
      selectObject(pos, ctrlDown, altDown);
      break;
    case TouchTapAction::Context:
      // if Ctrl down, show context menu (for all selected objects)
      if(ctrlDown)
        MainWindow::getInstance()->execPropertyMenu();
      else
        selectObjectAndShowContextMenu(pos, altDown);
      break;
  }
}

void MyTouchWidget::touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch doubletap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  openPropertyDialog(pos);
}

void MyTouchWidget::touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch longTap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  bool ctrlDown=modifiers & Qt::ControlModifier;
  bool altDown=modifiers & Qt::AltModifier;
  switch(touchLongTapAction) {
    case TouchTapAction::Select:
      selectObject(pos, ctrlDown, altDown);
      break;
    case TouchTapAction::Context:
      // if Ctrl down, show context menu (for all selected objects)
      if(ctrlDown)
        MainWindow::getInstance()->execPropertyMenu();
      else
        selectObjectAndShowContextMenu(pos, altDown);
      break;
  }
}

void MyTouchWidget::touchMoveSave1(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movesave1"<<endl;)
  switch(touchMove1Action) {
    case TouchMoveAction::Rotate: rotateInit(); break;
    case TouchMoveAction::Translate: translateInit(); break;
  }
}

void MyTouchWidget::touchMoveSave2(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movesave2"<<endl;)
  switch(touchMove2Action) {
    case TouchMoveAction::Rotate: break;
    case TouchMoveAction::Translate: translateInit(); break;
  }
  zoomInit();
  rotateInit();
  touchMove2RotateInScreenPlane=0;
}

void MyTouchWidget::touchMoveReset1(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movereset1"<<endl;)
  switch(touchMove1Action) {
    case TouchMoveAction::Rotate: rotateReset(); break;
    case TouchMoveAction::Translate: translateReset(); break;
  }
}

void MyTouchWidget::touchMoveReset2(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movereset2"<<endl;)
  switch(touchMove2Action) {
    case TouchMoveAction::Rotate: break;
    case TouchMoveAction::Translate: translateReset(); break;
  }
  zoomReset();
  rotateReset();
}

void MyTouchWidget::touchMove1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG touch move1 rel="<<rel.x()<<" "<<rel.y()<<endl;)
  switch(touchMove1Action) {
    case TouchMoveAction::Rotate: rotateInScreenAxis(rel); break;
    case TouchMoveAction::Translate: translate(rel); break;
  }
}

void MyTouchWidget::touchMove2(Qt::KeyboardModifiers modifiers, const array<QPoint, 2> &initialPos, const array<QPoint, 2> &pos) {
  DEBUG(cout<<"DEBUG touch move2 initialPos="<<initialPos[0].x()<<" "<<initialPos[0].y()<<"    "<<initialPos[1].x()<<" "<<initialPos[1].y()<<endl;)
  DEBUG(cout<<"                         pos="<<pos  [0].x()<<" "<<pos  [0].y()<<"    "<<pos  [1].x()<<" "<<pos  [1].y()<<endl;)

  auto initialRel=initialPos[1]-initialPos[0];
  auto rel=pos[1]-pos[0];
  double initialAngle=atan2(initialRel.y(), initialRel.x());
  double angle=atan2(rel.y(), rel.x());
  double relAngle=angle-initialAngle;
  if(touchMove2RotateInScreenPlane==0) {
    bool ctrlDown=modifiers & Qt::ControlModifier;
    auto initialCenter=(initialPos[0]+initialPos[1])/2;
    auto center=(pos[0]+pos[1])/2;
    auto centerRel=center-initialCenter;
    switch(touchMove2Action) {
      case TouchMoveAction::Rotate: rotateInScreenAxis(centerRel); break;
      case TouchMoveAction::Translate: translate(centerRel); break;
    }

    auto initialDist=sqrt(QPoint::dotProduct(initialRel, initialRel));
    auto dist=sqrt(QPoint::dotProduct(rel, rel));
    int zoomValue=lround(dist-initialDist);
    if(!ctrlDown)
      zoomCameraAngle(zoomValue);
    else
      zoomCameraFocalDist(zoomValue);

    if(std::abs(relAngle)>inScreenRotateSwitch*M_PI/180) {
      touchMoveReset2(modifiers);
      touchMove2RotateInScreenPlane=relAngle;
    }
  }
  else
    rotateInScreenPlane(relAngle-touchMove2RotateInScreenPlane);
}



vector<pair<Body*, vector<SbVec3f>>> MyTouchWidget::getObjectsByRay(const QPoint &pos) {
  // get picked points by ray
  SoRayPickAction pickAction(MainWindow::getInstance()->glViewer->getViewportRegion());
  pickAction.setPoint(SbVec2s(pos.x(), MainWindow::getInstance()->glViewer->getViewportRegion().getViewportSizePixels()[1]-pos.y()));
  pickAction.setRadius(pickObjectRadius);
  pickAction.setPickAll(true);
  pickAction.apply(MainWindow::getInstance()->glViewer->getSceneManager()->getSceneGraph());
  auto pickedPoints=pickAction.getPickedPointList();
  vector<pair<Body*, vector<SbVec3f>>> ret;
  for(int i=0; pickedPoints[i]; i++) {
    SoPath *path=pickedPoints[i]->getPath();
    for(int j=path->getLength()-1; j>=0; j--) {
      auto it=Body::getBodyMap().find(path->getNode(j));
      if(it!=Body::getBodyMap().end()) {
        auto bodyIt=std::find_if(ret.begin(), ret.end(), [it](auto &bv) { return bv.first==it->second; });
        auto point=pickedPoints[i]->getPoint();
        if(bodyIt==ret.end())
          ret.emplace_back(it->second, vector<SbVec3f>(1, point));
        else
          bodyIt->second.emplace_back(point);

        // get picked point and delete the cameraPosition and cameraOrientation values (if camera moves with body)
        SbVec3f delta;
        MainWindow::getInstance()->cameraOrientation->inRotation.getValue().multVec(point, delta);
        float x, y, z;
        (delta+MainWindow::getInstance()->cameraPosition->vector[0]).getValue(x,y,z);
        QString str("Point [%1, %2, %3] on %4"); str=str.arg(x).arg(y).arg(z).arg(it->second->getObject()->getFullName().c_str());
        MainWindow::getInstance()->statusBar()->showMessage(str);
        fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<str.toStdString()<<endl;

        break;
      }
    }
  }
  return ret;
}

int MyTouchWidget::createObjectListMenu(const vector<Body*>& pickedObject) {
  QMenu menu(this);
  int ind=0;
  vector<Body*>::const_iterator it;
  for(it=pickedObject.begin(); it!=pickedObject.end(); it++) {
    auto *action=new QAction((*it)->icon(0),(*it)->getObject()->getFullName().c_str(),&menu);
    action->setData(QVariant(ind++));
    menu.addAction(action);
  }
  QAction *action=menu.exec(QCursor::pos());
  if(action!=nullptr)
    return action->data().toInt();
  return -1;
}

void MyTouchWidget::selectObject(const QPoint &pos, bool toggle, bool showMenuForAll) {
  MainWindow::getInstance()->objectList->setCurrentItem(nullptr);
  // select object
  // if toggle, toggle selection of object
  // if showMenuForAll and multiple objects are under the cursor, show a list of all these objects first and act on the selected then
  auto picked=getObjectsByRay(pos);
  if(picked.empty()) {
    MainWindow::getInstance()->objectSelected("", nullptr);
    return;
  }
  vector<Body*> bodies;
  transform(picked.begin(), picked.end(), back_inserter(bodies), [](auto &x){ return x.first; });
  int useObjIdx;
  if(picked.size()==1 || !showMenuForAll)
    useObjIdx=0;
  else {
    useObjIdx=createObjectListMenu(bodies);
    if(useObjIdx<0) {
      MainWindow::getInstance()->objectSelected("", nullptr);
      return;
    }
  }
  MainWindow::getInstance()->objectList->setCurrentItem(bodies[useObjIdx],0,
    toggle?QItemSelectionModel::Toggle:QItemSelectionModel::ClearAndSelect);
  MainWindow::getInstance()->objectSelected((bodies[useObjIdx])->getObject()->getID(), bodies[useObjIdx]);
}

void MyTouchWidget::selectObjectAndShowContextMenu(const QPoint &pos, bool showMenuForAll) {
  // if object is not selected, select object and show context menu
  // if object is selected, show context menu (for all selected objects)
  // if showMenuForAll and multiple objects are under the cursor, show a list of all these objects first and act on the selected then
  auto picked=getObjectsByRay(pos);
  if(picked.empty()) {
    MainWindow::getInstance()->objectSelected("", nullptr);
    return;
  }
  vector<Body*> bodies;
  transform(picked.begin(), picked.end(), back_inserter(bodies), [](auto &x){ return x.first; });
  int useObjIdx;
  if(picked.size()==1 || !showMenuForAll)
    useObjIdx=0;
  else {
    useObjIdx=createObjectListMenu(bodies);
    if(useObjIdx<0) {
      MainWindow::getInstance()->objectSelected("", nullptr);
      return;
    }
  }
  if(!MainWindow::getInstance()->objectList->selectedItems().contains(bodies[useObjIdx]))
    MainWindow::getInstance()->objectList->setCurrentItem(bodies[useObjIdx],0, QItemSelectionModel::ClearAndSelect);
  MainWindow::getInstance()->objectSelected((bodies[useObjIdx])->getObject()->getID(), bodies[useObjIdx]);
  auto *a=new QAction(Utils::QIconCached("seektopoint.svg"), "Seek view to point on this Body");
  auto body=bodies[useObjIdx];
  connect(a, &QAction::triggered,this, [this, pos, body](){
    seekToPoint(pos, body);
  });
  MainWindow::getInstance()->execPropertyMenu({a});
}

void MyTouchWidget::seekToPoint(const QPoint &pos, Body *body) {
  // seeks the focal point of the camera to the point on the shape under the cursor
  auto picked=getObjectsByRay(pos);
  if(picked.empty())
    return;
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  auto initialCameraPos=camera->position.getValue();
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  auto bvIt=picked.begin();
  if(body)
    bvIt=find_if(picked.begin(), picked.end(), [body](auto &x){ return x.first==body; });
  if(bvIt==picked.end())
    bvIt=picked.begin();
  SbVec3f midPoint(0,0,0);
  for(auto &r : bvIt->second)
    midPoint+=r;
  midPoint/=bvIt->second.size();
  auto cameraPos=midPoint+cameraVec*camera->focalDistance.getValue();
  MainWindow::getInstance()->startShortAni([camera, initialCameraPos, cameraPos](double c){
    camera->position.setValue(initialCameraPos + (cameraPos-initialCameraPos) * (0.5-0.5*cos(c*M_PI)));
  });
}

void MyTouchWidget::openPropertyDialog(const QPoint &pos) {
  auto *item=MainWindow::getInstance()->objectList->currentItem();
  if(!item && MainWindow::getInstance()->objectList->selectedItems().size()>0)
    item=MainWindow::getInstance()->objectList->selectedItems().first();
  if(!item)
    return;
  auto *object=static_cast<Object*>(item);
  // show properties dialog only if objectDoubleClicked is not connected to some other slot
  if(!MainWindow::getInstance()->isSignalConnected(QMetaMethod::fromSignal(&MainWindow::objectDoubleClicked)))
    object->getProperties()->openDialogSlot();
  MainWindow::getInstance()->objectDoubleClicked(object->getObject()->getID(), object);
}

void MyTouchWidget::rotateInit() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  initialRotateCameraOri=camera->orientation.getValue();
  initialRotateCameraPos=camera->position.getValue();
  SbMatrix oriMatrix;
  initialRotateCameraOri.getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  initialRotateCameraToPos=initialRotateCameraPos-camera->focalDistance.getValue()*cameraVec;
}

void MyTouchWidget::rotateReset() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  camera->orientation.setValue(initialRotateCameraOri);
  camera->position.setValue(initialRotateCameraPos);
}

void MyTouchWidget::rotateInScreenAxis(const QPoint &rel) {
  // rotate
//mfmf move dragger if D is pressed
//mfmf move dragger in constraint mode if Shift-D is pressed
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  // orientation
  SbVec3f axis(rel.y(), rel.x(), 0); // no - before rel.y() since rel.y() has a different sign then a 3D-Point
  initialRotateCameraOri.multVec(axis, axis);
  SbRotation relOri(axis, -sqrt(QPoint::dotProduct(rel, rel))*rotAnglePerPixel*M_PI/180);
  camera->orientation.setValue(initialRotateCameraOri*relOri);
  // position
  SbMatrix oriMatrix1;
  camera->orientation.getValue().getValue(oriMatrix1);
  SbVec3f cameraVec1(oriMatrix1[2][0], oriMatrix1[2][1], oriMatrix1[2][2]);
  auto cameraPos=initialRotateCameraToPos+camera->focalDistance.getValue()*cameraVec1;
  camera->position.setValue(cameraPos);
}

void MyTouchWidget::rotateInScreenPlane(double relAngle) {
  // if inScreenPlane, rotate in the screen plane
//mfmf move dragger if D is pressed
//mfmf move dragger in constraint mode if Shift-D is pressed
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  // orientation
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  SbRotation relOri(cameraVec, relAngle);
  camera->orientation.setValue(initialRotateCameraOri*relOri);
}

void MyTouchWidget::translateInit() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  initialTranslateCameraPos=camera->position.getValue();
}

void MyTouchWidget::translateReset() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  camera->position.setValue(initialTranslateCameraPos);
}

void MyTouchWidget::translate(const QPoint &rel) {
  // translate
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  const auto &cameraOri=camera->orientation.getValue();
  const auto &viewVolume=camera->getViewVolume();
  float z=camera->nearDistance.getValue()*(1-relCursorZ)+camera->farDistance.getValue()*relCursorZ;
  auto point3D00=viewVolume.getPlanePoint(z, SbVec2f(0,0));
  auto point3D11=viewVolume.getPlanePoint(z, SbVec2f(1,1));
  auto size3D=point3D11-point3D00;
  cameraOri.inverse().multVec(size3D, size3D);
  const auto &sizePixel=MainWindow::getInstance()->glViewer->getViewportRegion().getViewportSizePixels();
  float fac=max(size3D[0]/sizePixel[0], size3D[1]/sizePixel[1]);
  auto rel3D=SbVec3f(rel.x()*fac,-rel.y()*fac,0);
  cameraOri.multVec(rel3D, rel3D);
  camera->position.setValue(initialTranslateCameraPos-rel3D);
}

void MyTouchWidget::zoomInit() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  if(camera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    auto* orthCamera=static_cast<SoOrthographicCamera*>(camera);
    initialZoomCameraHeight=orthCamera->height.getValue();
  }
  else if(camera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    auto* persCamera=static_cast<SoPerspectiveCamera*>(camera);
    initialZoomCameraHeightAngle=persCamera->heightAngle.getValue();
  }
  initialZoomCameraPos=camera->position.getValue();;
  initialZoomCameraFocalDistance=camera->focalDistance.getValue();
}

void MyTouchWidget::zoomReset() {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  if(camera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    auto* orthCamera=static_cast<SoOrthographicCamera*>(camera);
    orthCamera->height.setValue(initialZoomCameraHeight);
  }
  else if(camera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    auto* persCamera=static_cast<SoPerspectiveCamera*>(camera);
    persCamera->heightAngle.setValue(initialZoomCameraHeightAngle);
  }
  camera->position.setValue(initialZoomCameraPos);
  camera->focalDistance.setValue(initialZoomCameraFocalDistance);
}

void MyTouchWidget::zoomCameraAngle(int change) {
  // zoom
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  float fac=pow(zoomFacPerPixel, -change);
  if(camera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    auto* orthCamera=static_cast<SoOrthographicCamera*>(camera);
    orthCamera->height.setValue(initialZoomCameraHeight*fac);
  }
  else if(camera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    auto* persCamera=static_cast<SoPerspectiveCamera*>(camera);
    float angle=initialZoomCameraHeightAngle*fac;
    persCamera->heightAngle.setValue(angle>M_PI ? M_PI : angle);
  }
}

void MyTouchWidget::zoomCameraFocalDist(int change) {
  // move the camera along the camera line while preserving the zoom (only relevant for perspectivec cameras)
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  float fac=pow(zoomFacPerPixel, -change);
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  float focalDistance=initialZoomCameraFocalDistance*fac;
  camera->focalDistance.setValue(focalDistance);
  auto toPoint=initialZoomCameraPos-cameraVec*initialZoomCameraFocalDistance;
  auto cameraPos=toPoint+cameraVec*focalDistance;
  camera->position.setValue(cameraPos);
  if(camera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    // nothing needed for orthographic cameras
  }
  else if(camera->getTypeId()==SoPerspectiveCamera::getClassTypeId()) {
    auto* persCamera=static_cast<SoPerspectiveCamera*>(camera);
    // adapt the heigthAngle to keep the zoom
    float x=tan(initialZoomCameraHeightAngle/2)*initialZoomCameraFocalDistance;
    float heightAngle=atan(x/focalDistance)*2;
    persCamera->heightAngle.setValue(heightAngle);
  }
}

void MyTouchWidget::changeFrame(int steps) {
  // change frame
  auto &frame=MainWindow::getInstance()->frame;
  frame->setValue(std::min(MainWindow::getInstance()->frameMaxSB->value(),
                  std::max(MainWindow::getInstance()->frameMinSB->value(),
                  static_cast<int>(frame->getValue())+steps)));
}

}
