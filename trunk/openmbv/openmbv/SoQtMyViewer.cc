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
#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoFaceSet.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoMaterial.h>
#include <QApplication>
#include "mainwindow.h"

SoQtMyViewer::SoQtMyViewer(QWidget *parent) : SoQtExaminerViewer(parent) {
  setDecoration(false);
  setTransparencyType(SoGLRenderAction::SORTED_OBJECT_BLEND);
  setAnimationEnabled(false);
  setSeekTime(1);
  setCameraType(SoOrthographicCamera::getClassTypeId());

  // background
  setClearBeforeRender(false); // clear by my background color
  bgSep=new SoSeparator;
  bgSep->ref();
  SoLightModel *l=new SoLightModel;
  bgSep->addChild(l);
  l->model.setValue(SoLightModel::BASE_COLOR);
  SoMaterialBinding *mb=new SoMaterialBinding;
  bgSep->addChild(mb);
  mb->value.setValue(SoMaterialBinding::PER_VERTEX);
  SoMaterial *m=new SoMaterial;
  bgSep->addChild(m);
  m->diffuseColor.connectFrom(MainWindow::getInstance()->getBgColor());
  SoCoordinate3 *c=new SoCoordinate3;
  bgSep->addChild(c);
  c->point.set1Value(0, -1, -1, 0);
  c->point.set1Value(1, +1, -1, 0);
  c->point.set1Value(2, +1, +1, 0);
  c->point.set1Value(3, -1, +1, 0);
  SoFaceSet *f=new SoFaceSet;
  bgSep->addChild(f);
  f->numVertices.setValue(4);

  // foreground
  fgSep=new SoSeparator;
  fgSep->ref();
  timeTrans=new SoTranslation;
  fgSep->addChild(timeTrans);
  SoBaseColor *soFgColorTop=new SoBaseColor;
  fgSep->addChild(soFgColorTop);
  soFgColorTop->rgb.connectFrom(MainWindow::getInstance()->getFgColorTop());
  fgSep->addChild(MainWindow::getInstance()->getTimeString());
  ombvTrans=new SoTranslation;
  fgSep->addChild(ombvTrans);
  SoText2 *text2=new SoText2;
  SoBaseColor *soFgColorBottom=new SoBaseColor;
  fgSep->addChild(soFgColorBottom);
  soFgColorBottom->rgb.connectFrom(MainWindow::getInstance()->getFgColorBottom());
  fgSep->addChild(text2);
  text2->string.setValue("OpenMBV [http://openmbv.berlios.de]");
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *const event) {
  // if D is down unset viewing
  if(event->isOfType(SoKeyboardEvent::getClassTypeId())) {
    SoKeyboardEvent *e=(SoKeyboardEvent*)event;
    if(e->getKey()==SoKeyboardEvent::D) {
      if(e->getState()==SoKeyboardEvent::DOWN)
        setViewing(false);
      else if(e->getState()==SoKeyboardEvent::UP)
        setViewing(true);
      return true;
    }
  }
  
  if(MainWindow::getInstance()->soQtEventCB(event))
    return true;
  else
    return SoQtExaminerViewer::processSoEvent(event);
}

void SoQtMyViewer::actualRedraw(void) {
  // background
  getGLRenderAction()->apply(bgSep);

  // draw scene
  SoQtExaminerViewer::actualRedraw();

  // foreground (time and OpenMBV)
  glClear(GL_DEPTH_BUFFER_BIT);
  short x, y;
  getViewportRegion().getWindowSize().getValue(x, y);
  timeTrans->translation.setValue(-1+2.0/x*3,1-2.0/y*15,0);
  ombvTrans->translation.setValue(0,-1+2.0/y*15 -1+2.0/y*3,0);
  getGLRenderAction()->apply(fgSep);

  // update fps
  MainWindow::getInstance()->fpsCB();
}
