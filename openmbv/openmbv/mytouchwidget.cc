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
#include "openmbvcppinterface/group.h"
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
  setCursor3D(appSettings->get<bool>(AppSettings::mouseCursor3D));
  setLongTapInterval(appSettings->get<int>(AppSettings::tapAndHoldTimeout));

  auto set=[](auto &action, Modifier mod, AppSettings::AS appSet){
    action[mod]=static_cast<std::decay_t<decltype(action[mod])>>(appSettings->get<int>(appSet));
  };

  set(mouseLeftClickAction, Modifier::None , AppSettings::mouseNoneLeftClickAction);
  set(mouseLeftClickAction, Modifier::Shift, AppSettings::mouseShiftLeftClickAction);
  set(mouseLeftClickAction, Modifier::Ctrl, AppSettings::mouseCtrlLeftClickAction);
  set(mouseLeftClickAction, Modifier::Alt, AppSettings::mouseAltLeftClickAction);
  set(mouseLeftClickAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlLeftClickAction);
  set(mouseLeftClickAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltLeftClickAction);
  set(mouseLeftClickAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltLeftClickAction);
  set(mouseLeftClickAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltLeftClickAction);

  set(mouseRightClickAction, Modifier::None , AppSettings::mouseNoneRightClickAction);
  set(mouseRightClickAction, Modifier::Shift, AppSettings::mouseShiftRightClickAction);
  set(mouseRightClickAction, Modifier::Ctrl, AppSettings::mouseCtrlRightClickAction);
  set(mouseRightClickAction, Modifier::Alt, AppSettings::mouseAltRightClickAction);
  set(mouseRightClickAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlRightClickAction);
  set(mouseRightClickAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltRightClickAction);
  set(mouseRightClickAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltRightClickAction);
  set(mouseRightClickAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltRightClickAction);

  set(mouseMidClickAction, Modifier::None , AppSettings::mouseNoneMidClickAction);
  set(mouseMidClickAction, Modifier::Shift, AppSettings::mouseShiftMidClickAction);
  set(mouseMidClickAction, Modifier::Ctrl, AppSettings::mouseCtrlMidClickAction);
  set(mouseMidClickAction, Modifier::Alt, AppSettings::mouseAltMidClickAction);
  set(mouseMidClickAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlMidClickAction);
  set(mouseMidClickAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltMidClickAction);
  set(mouseMidClickAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltMidClickAction);
  set(mouseMidClickAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltMidClickAction);

  set(mouseLeftMoveAction, Modifier::None , AppSettings::mouseNoneLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::Shift, AppSettings::mouseShiftLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::Ctrl, AppSettings::mouseCtrlLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::Alt, AppSettings::mouseAltLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltLeftMoveAction);
  set(mouseLeftMoveAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltLeftMoveAction);

  set(mouseRightMoveAction, Modifier::None , AppSettings::mouseNoneRightMoveAction);
  set(mouseRightMoveAction, Modifier::Shift, AppSettings::mouseShiftRightMoveAction);
  set(mouseRightMoveAction, Modifier::Ctrl, AppSettings::mouseCtrlRightMoveAction);
  set(mouseRightMoveAction, Modifier::Alt, AppSettings::mouseAltRightMoveAction);
  set(mouseRightMoveAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlRightMoveAction);
  set(mouseRightMoveAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltRightMoveAction);
  set(mouseRightMoveAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltRightMoveAction);
  set(mouseRightMoveAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltRightMoveAction);

  set(mouseMidMoveAction, Modifier::None , AppSettings::mouseNoneMidMoveAction);
  set(mouseMidMoveAction, Modifier::Shift, AppSettings::mouseShiftMidMoveAction);
  set(mouseMidMoveAction, Modifier::Ctrl, AppSettings::mouseCtrlMidMoveAction);
  set(mouseMidMoveAction, Modifier::Alt, AppSettings::mouseAltMidMoveAction);
  set(mouseMidMoveAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlMidMoveAction);
  set(mouseMidMoveAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltMidMoveAction);
  set(mouseMidMoveAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltMidMoveAction);
  set(mouseMidMoveAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltMidMoveAction);

  set(mouseWheelAction, Modifier::None , AppSettings::mouseNoneWheelAction);
  set(mouseWheelAction, Modifier::Shift, AppSettings::mouseShiftWheelAction);
  set(mouseWheelAction, Modifier::Ctrl, AppSettings::mouseCtrlWheelAction);
  set(mouseWheelAction, Modifier::Alt, AppSettings::mouseAltWheelAction);
  set(mouseWheelAction, Modifier::ShiftCtrl, AppSettings::mouseShiftCtrlWheelAction);
  set(mouseWheelAction, Modifier::ShiftAlt, AppSettings::mouseShiftAltWheelAction);
  set(mouseWheelAction, Modifier::CtrlAlt, AppSettings::mouseCtrlAltWheelAction);
  set(mouseWheelAction, Modifier::ShiftCtrlAlt, AppSettings::mouseShiftCtrlAltWheelAction);

  set(touchTapAction, Modifier::None , AppSettings::touchNoneTapAction);
  set(touchTapAction, Modifier::Shift, AppSettings::touchShiftTapAction);
  set(touchTapAction, Modifier::Ctrl, AppSettings::touchCtrlTapAction);
  set(touchTapAction, Modifier::Alt, AppSettings::touchAltTapAction);
  set(touchTapAction, Modifier::ShiftCtrl, AppSettings::touchShiftCtrlTapAction);
  set(touchTapAction, Modifier::ShiftAlt, AppSettings::touchShiftAltTapAction);
  set(touchTapAction, Modifier::CtrlAlt, AppSettings::touchCtrlAltTapAction);
  set(touchTapAction, Modifier::ShiftCtrlAlt, AppSettings::touchShiftCtrlAltTapAction);

  set(touchLongTapAction, Modifier::None , AppSettings::touchNoneLongTapAction);
  set(touchLongTapAction, Modifier::Shift, AppSettings::touchShiftLongTapAction);
  set(touchLongTapAction, Modifier::Ctrl, AppSettings::touchCtrlLongTapAction);
  set(touchLongTapAction, Modifier::Alt, AppSettings::touchAltLongTapAction);
  set(touchLongTapAction, Modifier::ShiftCtrl, AppSettings::touchShiftCtrlLongTapAction);
  set(touchLongTapAction, Modifier::ShiftAlt, AppSettings::touchShiftAltLongTapAction);
  set(touchLongTapAction, Modifier::CtrlAlt, AppSettings::touchCtrlAltLongTapAction);
  set(touchLongTapAction, Modifier::ShiftCtrlAlt, AppSettings::touchShiftCtrlAltLongTapAction);

  set(touchMove1Action, Modifier::None , AppSettings::touchNoneMove1Action);
  set(touchMove1Action, Modifier::Shift, AppSettings::touchShiftMove1Action);
  set(touchMove1Action, Modifier::Ctrl, AppSettings::touchCtrlMove1Action);
  set(touchMove1Action, Modifier::Alt, AppSettings::touchAltMove1Action);
  set(touchMove1Action, Modifier::ShiftCtrl, AppSettings::touchShiftCtrlMove1Action);
  set(touchMove1Action, Modifier::ShiftAlt, AppSettings::touchShiftAltMove1Action);
  set(touchMove1Action, Modifier::CtrlAlt, AppSettings::touchCtrlAltMove1Action);
  set(touchMove1Action, Modifier::ShiftCtrlAlt, AppSettings::touchShiftCtrlAltMove1Action);

  set(touchMove2Action, Modifier::None , AppSettings::touchNoneMove2Action);
  set(touchMove2Action, Modifier::Shift, AppSettings::touchShiftMove2Action);
  set(touchMove2Action, Modifier::Ctrl, AppSettings::touchCtrlMove2Action);
  set(touchMove2Action, Modifier::Alt, AppSettings::touchAltMove2Action);
  set(touchMove2Action, Modifier::ShiftCtrl, AppSettings::touchShiftCtrlMove2Action);
  set(touchMove2Action, Modifier::ShiftAlt, AppSettings::touchShiftAltMove2Action);
  set(touchMove2Action, Modifier::CtrlAlt, AppSettings::touchCtrlAltMove2Action);
  set(touchMove2Action, Modifier::ShiftCtrlAlt, AppSettings::touchShiftCtrlAltMove2Action);

  set(touchMove2ZoomAction, Modifier::None , AppSettings::touchNoneMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::Shift, AppSettings::touchShiftMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::Ctrl, AppSettings::touchCtrlMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::Alt, AppSettings::touchAltMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::ShiftCtrl, AppSettings::touchShiftCtrlMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::ShiftAlt, AppSettings::touchShiftAltMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::CtrlAlt, AppSettings::touchCtrlAltMove2ZoomAction);
  set(touchMove2ZoomAction, Modifier::ShiftCtrlAlt, AppSettings::touchShiftCtrlAltMove2ZoomAction);

  zoomFacPerPixel=appSettings->get<double>(AppSettings::zoomFacPerPixel);
  zoomFacPerAngle=appSettings->get<double>(AppSettings::zoomFacPerAngle);
  rotAnglePerPixel=appSettings->get<double>(AppSettings::rotAnglePerPixel);
  pickObjectRadius=appSettings->get<double>(AppSettings::pickObjectRadius);
  inScreenRotateSwitch=appSettings->get<double>(AppSettings::inScreenRotateSwitch);
  relCursorZPerWheel=appSettings->get<double>(AppSettings::relCursorZPerWheel);
  relCursorZPerPixel=appSettings->get<double>(AppSettings::relCursorZPerPixel);
  pixelPerFrame=appSettings->get<double>(AppSettings::pixelPerFrame);
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
    case QEvent::TouchBegin:
    case QEvent::TouchUpdate:
    case QEvent::TouchEnd: {
      auto touchEvent=static_cast<QTouchEvent*>(event);
      if(touchEvent->touchPoints().size()==1)
        updateCursorPos(touchEvent->touchPoints()[0].pos().toPoint());
      if(touchEvent->touchPoints().size()==2)
        updateCursorPos((touchEvent->touchPoints()[0].pos().toPoint()+touchEvent->touchPoints()[1].pos().toPoint())/2);
    }
    default: break;
  }
  return TouchWidget<QWidget>::event(event);
}



namespace {
  MyTouchWidget::Modifier fromQtMod(Qt::KeyboardModifiers modifieres) {
    if(modifieres & Qt::ShiftModifier && modifieres & Qt::ControlModifier && modifieres & Qt::AltModifier)
      return MyTouchWidget::Modifier::ShiftCtrlAlt;
    if(modifieres & Qt::ShiftModifier && modifieres & Qt::ControlModifier)
      return MyTouchWidget::Modifier::ShiftCtrl;
    if(modifieres & Qt::ShiftModifier && modifieres & Qt::AltModifier)
      return MyTouchWidget::Modifier::ShiftAlt;
    if(modifieres & Qt::ControlModifier && modifieres & Qt::AltModifier)
      return MyTouchWidget::Modifier::CtrlAlt;
    if(modifieres & Qt::ShiftModifier)
      return MyTouchWidget::Modifier::Shift;
    if(modifieres & Qt::ControlModifier)
      return MyTouchWidget::Modifier::Ctrl;
    if(modifieres & Qt::AltModifier)
      return MyTouchWidget::Modifier::Alt;
    return MyTouchWidget::Modifier::None;
  }
}

void MyTouchWidget::mouseLeftClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse leftclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  switch(mouseLeftClickAction[fromQtMod(modifiers)]) {
    case ClickTapAction::None: break;
    case ClickTapAction::SelectTopObject: selectObject(pos, false, false); break;
    case ClickTapAction::ToggleTopObject: selectObject(pos, true, false); break;
    case ClickTapAction::SelectAnyObject: selectObject(pos, false, true); break;
    case ClickTapAction::ToggleAnyObject: selectObject(pos, true, true); break;
    case ClickTapAction::ShowContextMenu: MainWindow::getInstance()->execPropertyMenu(); break;
    case ClickTapAction::SelectTopObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, false); break;
    case ClickTapAction::SelectAnyObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, true); break;
    case ClickTapAction::SetRotationPointAndCursorSz: setRotationPointAndCursorSz(pos); break;
  }
}

void MyTouchWidget::mouseRightClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse rightclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  switch(mouseRightClickAction[fromQtMod(modifiers)]) {
    case ClickTapAction::None: break;
    case ClickTapAction::SelectTopObject: selectObject(pos, false, false); break;
    case ClickTapAction::ToggleTopObject: selectObject(pos, true, false); break;
    case ClickTapAction::SelectAnyObject: selectObject(pos, false, true); break;
    case ClickTapAction::ToggleAnyObject: selectObject(pos, true, true); break;
    case ClickTapAction::ShowContextMenu: MainWindow::getInstance()->execPropertyMenu(); break;
    case ClickTapAction::SelectTopObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, false); break;
    case ClickTapAction::SelectAnyObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, true); break;
    case ClickTapAction::SetRotationPointAndCursorSz: setRotationPointAndCursorSz(pos); break;
  }
}

void MyTouchWidget::mouseMidClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse midclick abs="<<pos.x()<<" "<<pos.y()<<endl;)
  switch(mouseMidClickAction[fromQtMod(modifiers)]) {
    case ClickTapAction::None: break;
    case ClickTapAction::SelectTopObject: selectObject(pos, false, false); break;
    case ClickTapAction::ToggleTopObject: selectObject(pos, true, false); break;
    case ClickTapAction::SelectAnyObject: selectObject(pos, false, true); break;
    case ClickTapAction::ToggleAnyObject: selectObject(pos, true, true); break;
    case ClickTapAction::ShowContextMenu: MainWindow::getInstance()->execPropertyMenu(); break;
    case ClickTapAction::SelectTopObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, false); break;
    case ClickTapAction::SelectAnyObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, true); break;
    case ClickTapAction::SetRotationPointAndCursorSz: setRotationPointAndCursorSz(pos); break;
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

void MyTouchWidget::mouseLeftMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {
  DEBUG(cout<<"DEBUG mouse leftsave"<<endl;)
  switch(mouseLeftMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateInit(initialPos); break;
    case MoveAction::Zoom: zoomInit(); break;
    case MoveAction::CameraAngle: zoomInit(); break;
    case MoveAction::CameraAndRotationPointSz: zoomInit(); break;
    case MoveAction::CameraNearPlane: zoomInit(); break;
    case MoveAction::CursorSz: zoomInit(); break;
    case MoveAction::RotateAboutSySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWxSx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWzSx: rotateInit(initialPos); break;
    case MoveAction::Translate: translateInit(); break;
  }
  initialFrame=MainWindow::getInstance()->frameNode->index[0];
}

void MyTouchWidget::mouseRightMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {
  DEBUG(cout<<"DEBUG mouse rightsave"<<endl;)
  switch(mouseRightMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateInit(initialPos); break;
    case MoveAction::Zoom: zoomInit(); break;
    case MoveAction::CameraAngle: zoomInit(); break;
    case MoveAction::CameraAndRotationPointSz: zoomInit(); break;
    case MoveAction::CameraNearPlane: zoomInit(); break;
    case MoveAction::CursorSz: zoomInit(); break;
    case MoveAction::RotateAboutSySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWxSx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWzSx: rotateInit(initialPos); break;
    case MoveAction::Translate: translateInit(); break;
  }
  initialFrame=MainWindow::getInstance()->frameNode->index[0];
}

void MyTouchWidget::mouseMidMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {
  DEBUG(cout<<"DEBUG mouse midsave"<<endl;)
  switch(mouseMidMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateInit(initialPos); break;
    case MoveAction::Zoom: zoomInit(); break;
    case MoveAction::CameraAngle: zoomInit(); break;
    case MoveAction::CameraAndRotationPointSz: zoomInit(); break;
    case MoveAction::CameraNearPlane: zoomInit(); break;
    case MoveAction::CursorSz: zoomInit(); break;
    case MoveAction::RotateAboutSySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWxSx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWzSx: rotateInit(initialPos); break;
    case MoveAction::Translate: translateInit(); break;
  }
  initialFrame=MainWindow::getInstance()->frameNode->index[0];
}

void MyTouchWidget::mouseLeftMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse leftReset"<<endl;)
  //rotateReset();// mfmf is this really needed?????
  switch(mouseLeftMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateReset(); break;
    case MoveAction::Zoom: zoomReset(); break;
    case MoveAction::CameraAngle: zoomReset(); break;
    case MoveAction::CameraAndRotationPointSz: zoomReset(); break;
    case MoveAction::CameraNearPlane: zoomReset(); break;
    case MoveAction::CursorSz: zoomReset(); break;
    case MoveAction::RotateAboutSySx: rotateReset(); break;
    case MoveAction::RotateAboutWxSx: rotateReset(); break;
    case MoveAction::RotateAboutWySx: rotateReset(); break;
    case MoveAction::RotateAboutWzSx: rotateReset(); break;
    case MoveAction::Translate: translateReset(); break;
  }
}

void MyTouchWidget::mouseRightMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse rightReset"<<endl;)
  switch(mouseRightMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateReset(); break;
    case MoveAction::Zoom: zoomReset(); break;
    case MoveAction::CameraAngle: zoomReset(); break;
    case MoveAction::CameraAndRotationPointSz: zoomReset(); break;
    case MoveAction::CameraNearPlane: zoomReset(); break;
    case MoveAction::CursorSz: zoomReset(); break;
    case MoveAction::RotateAboutSySx: rotateReset(); break;
    case MoveAction::RotateAboutWxSx: rotateReset(); break;
    case MoveAction::RotateAboutWySx: rotateReset(); break;
    case MoveAction::RotateAboutWzSx: rotateReset(); break;
    case MoveAction::Translate: translateReset(); break;
  }
}

void MyTouchWidget::mouseMidMoveReset(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG mouse midReset"<<endl;)
  switch(mouseMidMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: rotateReset(); break;
    case MoveAction::Zoom: zoomReset(); break;
    case MoveAction::CameraAngle: zoomReset(); break;
    case MoveAction::CameraAndRotationPointSz: zoomReset(); break;
    case MoveAction::CameraNearPlane: zoomReset(); break;
    case MoveAction::CursorSz: zoomReset(); break;
    case MoveAction::RotateAboutSySx: rotateReset(); break;
    case MoveAction::RotateAboutWxSx: rotateReset(); break;
    case MoveAction::RotateAboutWySx: rotateReset(); break;
    case MoveAction::RotateAboutWzSx: rotateReset(); break;
    case MoveAction::Translate: translateReset(); break;
  }
}

void MyTouchWidget::mouseLeftMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse leftMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  switch(mouseLeftMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: changeFrame(initialFrame-lround(float(rel.y())/pixelPerFrame), false); break;
    case MoveAction::RotateAboutSz: rotateAboutSz(rel.y()*rotAnglePerPixel*M_PI/180 ); break;
    case MoveAction::Zoom: zoom(-rel.y(), NOf); break;
    case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(rel.y()); break;
    case MoveAction::CameraAndRotationPointSz: cameraAndRotationPointSz(rel, pos); break;
    case MoveAction::CameraNearPlane: cameraNearPlane(rel, pos); break;
    case MoveAction::CursorSz: cursorSz(rel.y(), NOf, pos); break;
    case MoveAction::RotateAboutSySx: rotateAboutSySx(rel, pos); break;
    case MoveAction::RotateAboutWxSx: rotateAboutWxSx(rel, pos); break;
    case MoveAction::RotateAboutWySx: rotateAboutWySx(rel, pos); break;
    case MoveAction::RotateAboutWzSx: rotateAboutWzSx(rel, pos); break;
    case MoveAction::Translate: translate(rel); break;
  }
}

void MyTouchWidget::mouseRightMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse rightMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  switch(mouseRightMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: changeFrame(initialFrame-lround(float(rel.y())/pixelPerFrame), false); break;
    case MoveAction::RotateAboutSz: rotateAboutSz(rel.y()*rotAnglePerPixel*M_PI/180 ); break;
    case MoveAction::Zoom: zoom(-rel.y(), NOf); break;
    case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(rel.y()); break;
    case MoveAction::CameraAndRotationPointSz: cameraAndRotationPointSz(rel, pos); break;
    case MoveAction::CameraNearPlane: cameraNearPlane(rel, pos); break;
    case MoveAction::CursorSz: cursorSz(rel.y(), NOf, pos); break;
    case MoveAction::RotateAboutSySx: rotateAboutSySx(rel, pos); break;
    case MoveAction::RotateAboutWxSx: rotateAboutWxSx(rel, pos); break;
    case MoveAction::RotateAboutWySx: rotateAboutWySx(rel, pos); break;
    case MoveAction::RotateAboutWzSx: rotateAboutWzSx(rel, pos); break;
    case MoveAction::Translate: translate(rel); break;
  }
}

void MyTouchWidget::mouseMidMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG mouse midMove rel="<<rel.x()<<" "<<rel.y()<<endl;)
  switch(mouseMidMoveAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: changeFrame(initialFrame-lround(float(rel.y())/pixelPerFrame), false); break;
    case MoveAction::RotateAboutSz: rotateAboutSz(rel.y()*rotAnglePerPixel*M_PI/180 ); break;
    case MoveAction::Zoom: zoom(-rel.y(), NOf); break;
    case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(rel.y()); break;
    case MoveAction::CameraAndRotationPointSz: cameraAndRotationPointSz(rel, pos); break;
    case MoveAction::CameraNearPlane: cameraNearPlane(rel, pos); break;
    case MoveAction::CursorSz: cursorSz(rel.y(), NOf, pos); break;
    case MoveAction::RotateAboutSySx: rotateAboutSySx(rel, pos); break;
    case MoveAction::RotateAboutWxSx: rotateAboutWxSx(rel, pos); break;
    case MoveAction::RotateAboutWySx: rotateAboutWySx(rel, pos); break;
    case MoveAction::RotateAboutWzSx: rotateAboutWzSx(rel, pos); break;
    case MoveAction::Translate: translate(rel); break;
  }
}

void MyTouchWidget::mouseWheel(Qt::KeyboardModifiers modifiers, double relAngle, const QPoint &pos) {
  DEBUG(cout<<"DEBUG mouse wheel deltaAngle="<<relAngle<<"°"<<endl;)
  switch(mouseWheelAction[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: changeFrame(lround(relAngle/15)); break;
    case MoveAction::RotateAboutSz: rotateAboutSz(-relAngle*M_PI/180, false); break;
    case MoveAction::Zoom: zoomInit(); zoom(NOi, relAngle); break;
    case MoveAction::CameraAngle: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::CameraAndRotationPointSz: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::CameraNearPlane: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::CursorSz: zoomInit(); cursorSz(NOi, relAngle, pos); break;
    case MoveAction::RotateAboutSySx: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::RotateAboutWxSx: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::RotateAboutWySx: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::RotateAboutWzSx: throw runtime_error("Invalid move action for mouse wheel event."); break;
    case MoveAction::Translate: throw runtime_error("Invalid move action for mouse wheel event."); break;
  }
}



void MyTouchWidget::touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch tap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  switch(touchTapAction[fromQtMod(modifiers)]) {
    case ClickTapAction::None: break;
    case ClickTapAction::SelectTopObject: selectObject(pos, false, false); break;
    case ClickTapAction::ToggleTopObject: selectObject(pos, true, false); break;
    case ClickTapAction::SelectAnyObject: selectObject(pos, false, true); break;
    case ClickTapAction::ToggleAnyObject: selectObject(pos, true, true); break;
    case ClickTapAction::ShowContextMenu: MainWindow::getInstance()->execPropertyMenu(); break;
    case ClickTapAction::SelectTopObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, false); break;
    case ClickTapAction::SelectAnyObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, true); break;
    case ClickTapAction::SetRotationPointAndCursorSz: setRotationPointAndCursorSz(pos); break;
  }
}

void MyTouchWidget::touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch doubletap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  openPropertyDialog(pos);
}

void MyTouchWidget::touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  DEBUG(cout<<"DEBUG touch longTap pos="<<pos.x()<<" "<<pos.y()<<endl;)
  switch(touchLongTapAction[fromQtMod(modifiers)]) {
    case ClickTapAction::None: break;
    case ClickTapAction::SelectTopObject: selectObject(pos, false, false); break;
    case ClickTapAction::ToggleTopObject: selectObject(pos, true, false); break;
    case ClickTapAction::SelectAnyObject: selectObject(pos, false, true); break;
    case ClickTapAction::ToggleAnyObject: selectObject(pos, true, true); break;
    case ClickTapAction::ShowContextMenu: MainWindow::getInstance()->execPropertyMenu(); break;
    case ClickTapAction::SelectTopObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, false); break;
    case ClickTapAction::SelectAnyObjectAndShowContextMenu: selectObjectAndShowContextMenu(pos, true); break;
    case ClickTapAction::SetRotationPointAndCursorSz: setRotationPointAndCursorSz(pos); break;
  }
}

void MyTouchWidget::touchMoveSave1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {
  DEBUG(cout<<"DEBUG touch movesave1"<<endl;)
  switch(touchMove1Action[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::CameraAngle: zoomInit(); break;
    case MoveAction::CameraAndRotationPointSz: zoomInit(); break;
    case MoveAction::CameraNearPlane: zoomInit(); break;
    case MoveAction::CursorSz: zoomInit(); break;
    case MoveAction::RotateAboutSySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWxSx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWySx: rotateInit(initialPos); break;
    case MoveAction::RotateAboutWzSx: rotateInit(initialPos); break;
    case MoveAction::Translate: translateInit(); break;
  }
}

void MyTouchWidget::touchMoveSave2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos) {
  DEBUG(cout<<"DEBUG touch movesave2"<<endl;)
  switch(touchMove2Action[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
    case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
    case MoveAction::CameraAngle: zoomInit(); break;
    case MoveAction::CameraAndRotationPointSz: zoomInit(); break;
    case MoveAction::CameraNearPlane: zoomInit(); break;
    case MoveAction::CursorSz: zoomInit(); break;
    case MoveAction::RotateAboutSySx: rotateInit((initialPos[0]+initialPos[1])/2); break;
    case MoveAction::RotateAboutWxSx: rotateInit((initialPos[0]+initialPos[1])/2); break;
    case MoveAction::RotateAboutWySx: rotateInit((initialPos[0]+initialPos[1])/2); break;
    case MoveAction::RotateAboutWzSx: rotateInit((initialPos[0]+initialPos[1])/2); break;
    case MoveAction::Translate: translateInit(); break;
  }
  zoomInit();
//  rotateInit(); mfmf is this needed????
  touchMove2RotateInScreenPlane=0;
}

void MyTouchWidget::touchMoveReset1(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movereset1"<<endl;)
  switch(touchMove1Action[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::CameraAngle: zoomReset(); break;
    case MoveAction::CameraAndRotationPointSz: zoomReset(); break;
    case MoveAction::CameraNearPlane: zoomReset(); break;
    case MoveAction::CursorSz: zoomReset(); break;
    case MoveAction::RotateAboutSySx: rotateReset(); break;
    case MoveAction::RotateAboutWxSx: rotateReset(); break;
    case MoveAction::RotateAboutWySx: rotateReset(); break;
    case MoveAction::RotateAboutWzSx: rotateReset(); break;
    case MoveAction::Translate: translateReset(); break;
  }
}

void MyTouchWidget::touchMoveReset2(Qt::KeyboardModifiers modifiers) {
  DEBUG(cout<<"DEBUG touch movereset2"<<endl;)
  switch(touchMove2Action[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: break;
    case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
    case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
    case MoveAction::CameraAngle: zoomReset(); break;
    case MoveAction::CameraAndRotationPointSz: zoomReset(); break;
    case MoveAction::CameraNearPlane: zoomReset(); break;
    case MoveAction::CursorSz: zoomReset(); break;
    case MoveAction::RotateAboutSySx: rotateReset(); break;
    case MoveAction::RotateAboutWxSx: rotateReset(); break;
    case MoveAction::RotateAboutWySx: rotateReset(); break;
    case MoveAction::RotateAboutWzSx: rotateReset(); break;
    case MoveAction::Translate: translateReset(); break;
  }
  zoomReset();
//  rotateReset(); mfmf is this really needed????
}

void MyTouchWidget::touchMove1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {
  const QPoint rel=pos-initialPos;
  DEBUG(cout<<"DEBUG touch move1 rel="<<rel.x()<<" "<<rel.y()<<endl;)
  switch(touchMove1Action[fromQtMod(modifiers)]) {
    case MoveAction::None: break;
    case MoveAction::ChangeFrame: changeFrame(initialFrame-lround(float(rel.y())/pixelPerFrame), false); break;
    case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 1 finger event."); break;
    case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(rel.y()); break;
    case MoveAction::CameraAndRotationPointSz: cameraAndRotationPointSz(rel, pos); break;
    case MoveAction::CameraNearPlane: cameraNearPlane(rel, pos); break;
    case MoveAction::CursorSz: cursorSz(rel.y(), NOf, pos); break;
    case MoveAction::RotateAboutSySx: rotateAboutSySx(rel, pos); break;
    case MoveAction::RotateAboutWxSx: rotateAboutWxSx(rel, pos); break;
    case MoveAction::RotateAboutWySx: rotateAboutWySx(rel, pos); break;
    case MoveAction::RotateAboutWzSx: rotateAboutWzSx(rel, pos); break;
    case MoveAction::Translate: translate(rel); break;
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
    auto initialCenter=(initialPos[0]+initialPos[1])/2;
    auto center=(pos[0]+pos[1])/2;
    auto centerRel=center-initialCenter;
    switch(touchMove2Action[fromQtMod(modifiers)]) {
      case MoveAction::None: break;
      case MoveAction::ChangeFrame: changeFrame(initialFrame-lround(float(rel.y())/pixelPerFrame), false); break;
      case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::Zoom: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(centerRel.y()); break;
      case MoveAction::CameraAndRotationPointSz: cameraAndRotationPointSz(centerRel, center); break;
      case MoveAction::CameraNearPlane: cameraNearPlane(centerRel, center); break;
      case MoveAction::CursorSz: cursorSz(rel.y(), NOf, centerRel); break;
      case MoveAction::RotateAboutSySx: rotateAboutSySx(centerRel, center); break;
      case MoveAction::RotateAboutWxSx: rotateAboutWxSx(centerRel, center); break;
      case MoveAction::RotateAboutWySx: rotateAboutWySx(centerRel, center); break;
      case MoveAction::RotateAboutWzSx: rotateAboutWzSx(centerRel, center); break;
      case MoveAction::Translate: translate(centerRel); break;
    }

    auto initialDist=sqrt(QPoint::dotProduct(initialRel, initialRel));
    auto dist=sqrt(QPoint::dotProduct(rel, rel));
    int zoomValue=dist-initialDist;
    switch(touchMove2ZoomAction[fromQtMod(modifiers)]) {
      case MoveAction::None: break;
      case MoveAction::ChangeFrame: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::RotateAboutSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::Zoom: zoom(-zoomValue, NOf); break;
      case MoveAction::CameraAngle: zoomPerspectiveCameraAngle(zoomValue); break;
      case MoveAction::CameraAndRotationPointSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::CameraNearPlane: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::CursorSz: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::RotateAboutSySx: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::RotateAboutWxSx: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::RotateAboutWySx: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::RotateAboutWzSx: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
      case MoveAction::Translate: throw runtime_error("Invalid move action for touch move 2 finger event."); break;
    }

    if(std::abs(relAngle)>inScreenRotateSwitch*M_PI/180) {
      touchMoveReset2(modifiers);
      touchMove2RotateInScreenPlane=relAngle;
    }
  }
  else
    rotateAboutSz(relAngle-touchMove2RotateInScreenPlane);
}



vector<pair<Body*, vector<SbVec3f>>> MyTouchWidget::getObjectsByRay(const QPoint &pos) {
  // get picked points by ray
  int x=pos.x();
  int y=pos.y();
  auto size=MainWindow::getInstance()->glViewer->getViewportRegion().getViewportSizePixels();
  if(MainWindow::getInstance()->dialogStereo) {
    if(MainWindow::getInstance()->glViewerWG==this)
      x=x/2;
    else
      x=(size[0]+x)/2;
  }
  SoRayPickAction pickAction(MainWindow::getInstance()->glViewer->getViewportRegion());
  pickAction.setPoint(SbVec2s(x, size[1]-y));
  pickAction.setRadius(pickObjectRadius);
  pickAction.setPickAll(true);
  MainWindow::getInstance()->pickUpdate();
  pickAction.apply(MainWindow::getInstance()->glViewer->getSceneManager()->getSceneGraph());
  MainWindow::getInstance()->pickUpdateRestore();
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
        fmatvec::Atom::msgStatic(fmatvec::Atom::Status)<<str.toStdString()<<endl;

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

SbVec3f MyTouchWidget::convertToRel3D(const QPoint &rel) {
  auto relCursorZ = MainWindow::getInstance()->relCursorZ->getValue();

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
  if(MainWindow::getInstance()->dialogStereo)
    fac/=2.0;
  return {rel.x()*fac,-rel.y()*fac,0};
}

void MyTouchWidget::selectObject(const QPoint &pos, bool toggle, bool showMenuForAll) {
  auto picked=getObjectsByRay(pos);
  if(picked.empty()) {
    // nothing picked -> disable selection and emit "no object selected"
    MainWindow::getInstance()->objectList->setCurrentItem(nullptr);
    MainWindow::getInstance()->objectSelected("", nullptr);
    return;
  }

  vector<Body*> bodies;
  transform(picked.begin(), picked.end(), back_inserter(bodies), [](auto &x){ return x.first; });

  if(true) {
    // do not use a body inside of a CompoundRigidBody
    for(auto & bodyi : bodies) {
      Object *body = bodyi;
      while(true) {
        auto parent = static_cast<Object*>(body->QTreeWidgetItem::parent());
        if(dynamic_cast<Group*>(parent))
          break;
        body = parent;
      }
      bodyi = static_cast<Body*>(body);
    }
    // make bodies unique
    std::sort(bodies.begin(), bodies.end());
    bodies.erase(std::unique(bodies.begin(), bodies.end()), bodies.end());
  }

  int useObjIdx;
  if(bodies.size()==1 || !showMenuForAll)
    // if only one object is picked or no menu to choose the object is requested -> use the first picked object ...
    useObjIdx=0;
  else {
    // ... else show a menu to choose the object ...
    useObjIdx=createObjectListMenu(bodies);
    if(useObjIdx<0) {
      // ... or disable the selection and emit "no object selected" if nothing is choosen
      MainWindow::getInstance()->objectList->setCurrentItem(nullptr);
      MainWindow::getInstance()->objectSelected("", nullptr);
      return;
    }
  }

  // if toggle is and only 1 object is selected and this object is picked then toggle anyway
  if(!toggle &&
     MainWindow::getInstance()->objectList->selectedItems().size()==1 &&
     MainWindow::getInstance()->objectList->selectedItems()[0] == bodies[useObjIdx]) {
    MainWindow::getInstance()->objectList->setCurrentItem(nullptr);
    MainWindow::getInstance()->objectSelected("", nullptr);
    return;
  }

  // select the choosen object and emit the selected object (by ID if it has as ID)
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
    setRotationPointAndCursorSz(pos, body);
  });
  MainWindow::getInstance()->execPropertyMenu({a});
}

void MyTouchWidget::setRotationPointAndCursorSz(const QPoint &pos, Body *body) {
  // set the rotation point of the scene to the clicked point (this is done using the camera focalDistance)
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
  auto initialCameraFocalDistance=camera->focalDistance.getValue();
  auto cameraFocalDistance=(midPoint-initialCameraPos).normalize(); // keep the camera at its screen-z position
  auto cameraPos=midPoint+cameraVec*cameraFocalDistance;
  MainWindow::getInstance()->startShortAni([camera, initialCameraPos, cameraPos, initialCameraFocalDistance, cameraFocalDistance](double c){
    camera->position.setValue(initialCameraPos + (cameraPos-initialCameraPos) * (0.5-0.5*cos(c*M_PI)));
    camera->focalDistance.setValue(initialCameraFocalDistance + (cameraFocalDistance-initialCameraFocalDistance) * (0.5-0.5*cos(c*M_PI)));
  });

  // set the relCursorZ position to the clicked point
  auto d = (camera->position.getValue()-midPoint).normalize();
  auto nd = camera->nearDistance.getValue();
  auto fd = camera->farDistance.getValue();
  MainWindow::getInstance()->relCursorZ->setValue((d - nd)/(fd - nd));
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

void MyTouchWidget::rotateInit(const QPoint &initialPos) {
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

void MyTouchWidget::rotateAboutSySx(const QPoint &rel, const QPoint &pos) {
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

void MyTouchWidget::rotateAboutWSx(const QPoint &rel, const QPoint &pos, int axisIdx) {
  // rotate
//mfmf move dragger if D is pressed
//mfmf move dragger in constraint mode if Shift-D is pressed
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  // orientation
  SbVec3f W_xS(1, 0, 0); initialRotateCameraOri.multVec(W_xS, W_xS);
  auto angle_W_xS=rel.y()*rotAnglePerPixel*M_PI/180;
  SbRotation T12(W_xS, -angle_W_xS);
  float v[3]={0,0,0};
  v[axisIdx]=1;
  SbVec3f W_zW(v);
  auto angle_W_zW=-rel.x()*rotAnglePerPixel*M_PI/180;
  SbRotation T23(W_zW, angle_W_zW);
  camera->orientation.setValue(initialRotateCameraOri*T12*T23);
  // position
  SbMatrix oriMatrix1;
  camera->orientation.getValue().getValue(oriMatrix1);
  SbVec3f cameraVec1(oriMatrix1[2][0], oriMatrix1[2][1], oriMatrix1[2][2]);
  auto cameraPos=initialRotateCameraToPos+camera->focalDistance.getValue()*cameraVec1;
  camera->position.setValue(cameraPos);
}

void MyTouchWidget::rotateAboutWxSx(const QPoint &rel, const QPoint &pos) {
  rotateAboutWSx(rel, pos, 0);
}

void MyTouchWidget::rotateAboutWySx(const QPoint &rel, const QPoint &pos) {
  rotateAboutWSx(rel, pos, 1);
}

void MyTouchWidget::rotateAboutWzSx(const QPoint &rel, const QPoint &pos) {
  rotateAboutWSx(rel, pos, 2);
}

void MyTouchWidget::rotateAboutSz(double relAngle, bool relToInitial) {
  // if inScreenPlane, rotate in the screen plane
//mfmf move dragger if D is pressed
//mfmf move dragger in constraint mode if Shift-D is pressed
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  // orientation
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  SbRotation relOri(cameraVec, relAngle);
  if(relToInitial)
    camera->orientation.setValue(initialRotateCameraOri*relOri);
  else
    camera->orientation.setValue(camera->orientation.getValue()*relOri);
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
  auto rel3D=convertToRel3D(rel);
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  const auto &cameraOri=camera->orientation.getValue();
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
  initialZoomCameraNearPlaneValue=MainWindow::getInstance()->getNearPlaneValue();
  initialZoomRelCursorZ=MainWindow::getInstance()->relCursorZ->getValue();
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
  MainWindow::getInstance()->setNearPlaneValue(initialZoomCameraNearPlaneValue);
  MainWindow::getInstance()->relCursorZ->setValue(initialZoomRelCursorZ);
  camera->focalDistance.setValue(initialZoomCameraFocalDistance);
}

void MyTouchWidget::zoom(int relPixel, float relAngle) {
  // zoom
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  if(camera->getTypeId()==SoOrthographicCamera::getClassTypeId()) {
    float fac = relPixel!=NOi ? pow(zoomFacPerPixel, relPixel) : pow(zoomFacPerAngle, relAngle);
    auto* orthCamera=static_cast<SoOrthographicCamera*>(camera);
    orthCamera->height.setValue(initialZoomCameraHeight*fac);
    MainWindow::getInstance()->statusBar()->
      showMessage(QString("Camera height: %1").arg(initialZoomCameraHeight*fac,0,'f',6), 1000);
  }
  else if(camera->getTypeId()==SoPerspectiveCamera::getClassTypeId())
    zoomPerspectiveCameraDistance(relPixel, relAngle);
}

void MyTouchWidget::zoomPerspectiveCameraAngle(int relPixel) {
  float fac = pow(zoomFacPerPixel, relPixel);
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  auto* persCamera=static_cast<SoPerspectiveCamera*>(camera);
  float angle=initialZoomCameraHeightAngle/fac;
  persCamera->heightAngle.setValue(angle>M_PI ? M_PI : angle);
  MainWindow::getInstance()->statusBar()->
    showMessage(QString("Camera angle: %1°").arg((angle>M_PI ? M_PI : angle)*180/M_PI,0,'f',2), 1000);
}

void MyTouchWidget::zoomPerspectiveCameraDistance(int relPixel, float relAngle) {
  float fac = relPixel!=NOi ? pow(zoomFacPerPixel, relPixel) : pow(zoomFacPerAngle, relAngle);
  // move the camera along the camera line while keeping the focal point (only relevant for perspectivec cameras)
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  float focalDistance=initialZoomCameraFocalDistance*fac;
  camera->focalDistance.setValue(focalDistance);
  MainWindow::getInstance()->statusBar()->
    showMessage(QString("Camera distance from focal-point: %1").arg(focalDistance,0,'f',6), 1000);
  auto toPoint=initialZoomCameraPos-cameraVec*initialZoomCameraFocalDistance;
  auto cameraPos=toPoint+cameraVec*focalDistance;
  camera->position.setValue(cameraPos);
}

void MyTouchWidget::cameraAndRotationPointSz(const QPoint &rel, const QPoint &pos) {
  float x,y,z;
  convertToRel3D(rel).getValue(x,y,z);
  auto rel3D=-y;
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  SbMatrix oriMatrix;
  camera->orientation.getValue().getValue(oriMatrix);
  SbVec3f cameraVec(oriMatrix[2][0], oriMatrix[2][1], oriMatrix[2][2]);
  auto cameraPos=initialZoomCameraPos-cameraVec*rel3D;
  camera->position.setValue(cameraPos);
  MainWindow::getInstance()->statusBar()->
    showMessage(QString("Camera and rotation point relative screen-z move: %1%2").arg(rel3D>=0?'+':'-').arg(abs(rel3D), 0, 'f', 6), 1000);
}

void MyTouchWidget::cameraNearPlane(const QPoint &rel, const QPoint &pos) {
  auto *camera=MainWindow::getInstance()->glViewer->getCamera();
  if(camera->getTypeId()!=SoPerspectiveCamera::getClassTypeId())
    return;

  static bool nearPlaneByDistance=getenv("OPENMBV_NEARPLANEBYDISTANCE")!=nullptr;
  if(nearPlaneByDistance) {
    float x,y,z;
    convertToRel3D(rel).getValue(x,y,z);
    auto rel3D=-y;
    float nearPlaneDistance=initialZoomCameraNearPlaneValue-rel3D;
    if(nearPlaneDistance<1e-6)
      nearPlaneDistance=1e-6;
    if(nearPlaneDistance>camera->farDistance.getValue()*0.999)
      nearPlaneDistance=camera->farDistance.getValue()*0.999;
    MainWindow::getInstance()->setNearPlaneValue(nearPlaneDistance);
    MainWindow::getInstance()->frameNode->index.touch(); // force rendering the scene
    MainWindow::getInstance()->statusBar()->
      showMessage(QString("Camera near clipping plane at screen-z: %2").arg(nearPlaneDistance, 0, 'f', 6), 1000);
  }
  else {
    auto rel01=static_cast<float>(rel.y())/size().height();
    float nearPlaneFactor=initialZoomCameraNearPlaneValue-rel01;
    if(nearPlaneFactor<0.1)
      nearPlaneFactor=0.1;
    if(nearPlaneFactor>0.9)
      nearPlaneFactor=0.9;
    MainWindow::getInstance()->setNearPlaneValue(nearPlaneFactor);
    MainWindow::getInstance()->frameNode->index.touch(); // force rendering the scene
    MainWindow::getInstance()->statusBar()->
      showMessage(QString("Camera near clipping plane normalized factor: %1").arg(nearPlaneFactor, 0, 'f', 6), 1000);
  }
}

void MyTouchWidget::cursorSz(int relPixel, float relAngle, const QPoint &pos) {
  auto rel01=relPixel!=NOi ? static_cast<float>(relPixel)/size().height() : relAngle/15*relCursorZPerWheel;
  float relCursorZ=initialZoomRelCursorZ-rel01;
  if(relCursorZ<0.001)
    relCursorZ=0.001;
  if(relCursorZ>0.999)
    relCursorZ=0.999;
  MainWindow::getInstance()->relCursorZ->setValue(relCursorZ);
  MainWindow::getInstance()->statusBar()->showMessage(QString("Cursor screen-z: %1 (0.0/1.0=near/far clipping plane)").
    arg(relCursorZ, 0, 'f', 3), 1000);
  updateCursorPos(pos);
}

void MyTouchWidget::changeFrame(int steps, bool rel) {
  // change frame
  auto &frame=MainWindow::getInstance()->frameNode->index;
  frame.setValue(std::min(MainWindow::getInstance()->frameMaxSB->value(),
                 std::max(MainWindow::getInstance()->frameMinSB->value(),
                 rel ? static_cast<int>(frame[0])+steps : steps)));
}

void MyTouchWidget::updateCursorPos(const QPoint &mousePos) {
  if(!cursor3D)
    return;

  auto relCursorZ = MainWindow::getInstance()->relCursorZ->getValue();

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
  SbVec3f nearDist, farDist;
  vv.projectPointToLine(SbVec2f(pos[0],pos[1]), nearDist, farDist);
  SbVec3f cursorPos;
  if(MainWindow::getInstance()->glViewer->getCamera()->getTypeId()==SoOrthographicCamera::getClassTypeId())
    cursorPos=nearDist*(1-0.5)+farDist*0.5;
  else
    cursorPos=nearDist*(1-relCursorZ)+farDist*relCursorZ;
  MainWindow::getInstance()->setCursorPos(&cursorPos);
}

void MyTouchWidget::setVerticalAxis(MoveAction act) {
  if(act==MoveAction::RotateAboutSySx || act==MoveAction::RotateAboutWzSx)
    verticalAxis=2;
  else if(act==MoveAction::RotateAboutWxSx)
    verticalAxis=0;
  else if(act==MoveAction::RotateAboutWySx)
    verticalAxis=1;
  if(act==MoveAction::RotateAboutWxSx || act==MoveAction::RotateAboutWySx || act==MoveAction::RotateAboutWzSx) {
    SbMatrix cameraOri;
    MainWindow::getInstance()->glViewer->getCamera()->orientation.getValue().getValue(cameraOri);
    SbVec3f SverticalAxis(cameraOri.getValue()[0][verticalAxis],
                          cameraOri.getValue()[1][verticalAxis],
                          cameraOri.getValue()[2][verticalAxis]);
    if(abs(SverticalAxis[1])>1e-7)
      rotateAboutSz(-atan(SverticalAxis[0]/SverticalAxis[1]));
    else if(abs(SverticalAxis[0])>1e-7)
      rotateAboutSz(-M_PI/2);
  }
}

void MyTouchWidget::setCursor3D(bool value) {
  cursor3D = value;
  setCursor(cursor3D ? Qt::BlankCursor : Qt::CrossCursor);
  if(!cursor3D)
    MainWindow::getInstance()->setCursorPos();
}

}
