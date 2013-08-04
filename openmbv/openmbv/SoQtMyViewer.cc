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
#include <Inventor/nodes/SoTexture2.h>
#include <Inventor/nodes/SoTextureCoordinate2.h>
#include <QApplication>
#include <QIcon>
#include "mainwindow.h"

using namespace std;

namespace OpenMBVGUI {

SoQtMyViewer::SoQtMyViewer(QWidget *parent, int transparency) : SoQtExaminerViewer(parent) {
  setDecoration(false);
  switch(transparency) {
    case 2:
      setAlphaChannel(true);
      setTransparencyType(SoGLRenderAction::SORTED_LAYERS_BLEND);
      break;
    case 1:
    default:
      setTransparencyType(SoGLRenderAction::DELAYED_BLEND);
      break;
  }
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
  // time (top left)
  timeTrans=new SoTranslation;
  fgSep->addChild(timeTrans);
  SoBaseColor *soFgColorTop=new SoBaseColor;
  fgSep->addChild(soFgColorTop);
  soFgColorTop->rgb.connectFrom(MainWindow::getInstance()->getFgColorTop());
  fgSep->addChild(MainWindow::getInstance()->getTimeString());
  // ombvText (bottom left)
  ombvTrans=new SoTranslation;
  fgSep->addChild(ombvTrans);
  SoText2 *text2=new SoText2;
  SoBaseColor *soFgColorBottom=new SoBaseColor;
  fgSep->addChild(soFgColorBottom);
  soFgColorBottom->rgb.connectFrom(MainWindow::getInstance()->getFgColorBottom());
  fgSep->addChild(text2);
  text2->string.setValue("OpenMBV [http://code.google.com/p/openmbv]");
  // ombvLogo (bottom right)
  ombvLogoTrans=new SoTranslation;
  fgSep->addChild(ombvLogoTrans);
  ombvLogoScale=new SoScale;
  fgSep->addChild(ombvLogoScale);
  SoMaterial *cc=new SoMaterial;
  fgSep->addChild(cc);
  cc->emissiveColor.setValue(1,1,1);
  cc->transparency.setValue(0.6);
  SoTexture2 *ombvLogoTex=new SoTexture2;
  fgSep->addChild(ombvLogoTex);
  QIcon icon=Utils::QIconCached(":/openmbv.svg");
  QImage image=icon.pixmap(100,100).toImage();
  ombvLogoTex->image.setValue(SbVec2s(image.width(), image.height()), 4, image.bits());
  ombvLogoTex->wrapS.setValue(SoTexture2::CLAMP);
  ombvLogoTex->wrapT.setValue(SoTexture2::CLAMP);
  SoCoordinate3 *ombvCoords=new SoCoordinate3;
  fgSep->addChild(ombvCoords);
  double size=0.15; // the logo filles maximal "size" of the screen
  ombvCoords->point.set1Value(0, 0, 0, 0);
  ombvCoords->point.set1Value(1, 0, size, 0);
  ombvCoords->point.set1Value(2, -size, size, 0);
  ombvCoords->point.set1Value(3, -size, 0, 0);
  SoTextureCoordinate2 *tc=new SoTextureCoordinate2;
  fgSep->addChild(tc);
  tc->point.set1Value(0, 1, 1);
  tc->point.set1Value(1, 1, 0);
  tc->point.set1Value(2, 0, 0);
  tc->point.set1Value(3, 0, 1);
  SoFaceSet *ombvLogo=new SoFaceSet;
  fgSep->addChild(ombvLogo);
  ombvLogo->numVertices.set1Value(0, 4);
}

SoQtMyViewer::~SoQtMyViewer() {
  fgSep->unref();
  bgSep->unref();
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
  ombvLogoTrans->translation.setValue(+1-2.0/x*3 +1-2.0/x*3, 0,0);
  ombvLogoScale->scaleFactor.setValue(x>y?(float)y/x:1,y>x?(float)x/y:1,1);
  getGLRenderAction()->apply(fgSep);

  // update fps
  MainWindow::getInstance()->fpsCB();
}

}
