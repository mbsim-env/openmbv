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
  auto *l=new SoLightModel;
  bgSep->addChild(l);
  l->model.setValue(SoLightModel::BASE_COLOR);
  auto *mb=new SoMaterialBinding;
  bgSep->addChild(mb);
  mb->value.setValue(SoMaterialBinding::PER_VERTEX);
  auto *m=new SoMaterial;
  bgSep->addChild(m);
  m->diffuseColor.connectFrom(MainWindow::getInstance()->getBgColor());
  auto *c=new SoCoordinate3;
  bgSep->addChild(c);
  c->point.set1Value(0, -1, -1, 0);
  c->point.set1Value(1, +1, -1, 0);
  c->point.set1Value(2, +1, +1, 0);
  c->point.set1Value(3, -1, +1, 0);
  auto *f=new SoFaceSet;
  bgSep->addChild(f);
  f->numVertices.setValue(4);

  // foreground
  fgSep=new SoSeparator;
  fgSep->ref();
  // font size
  font=new SoFont;
  fgSep->addChild(font);
  font->size.setValue(10);
  // time (top left)
  auto *timeSep=new SoSeparator;
  fgSep->addChild(timeSep);
  timeTrans=new SoTranslation;
  timeSep->addChild(timeTrans);
  auto *soFgColorTop=new SoBaseColor;
  timeSep->addChild(soFgColorTop);
  soFgColorTop->rgb.connectFrom(MainWindow::getInstance()->getFgColorTop());
  timeSep->addChild(MainWindow::getInstance()->getTimeString());
  // ombvText (bottom left)
  auto *ombvSep=new SoSeparator;
  fgSep->addChild(ombvSep);
  ombvTrans=new SoTranslation;
  ombvSep->addChild(ombvTrans);
  auto *text2=new SoText2;
  auto *soFgColorBottom=new SoBaseColor;
  ombvSep->addChild(soFgColorBottom);
  soFgColorBottom->rgb.connectFrom(MainWindow::getInstance()->getFgColorBottom());
  ombvSep->addChild(text2);
  text2->string.setValue("OpenMBV [https://www.mbsim-env.de/]");
  // ombvLogo (bottom right)
  auto *logoSep=new SoSeparator;
  fgSep->addChild(logoSep);
  ombvLogoTrans=new SoTranslation;
  logoSep->addChild(ombvLogoTrans);
  ombvLogoScale=new SoScale;
  logoSep->addChild(ombvLogoScale);
  auto *cc=new SoMaterial;
  logoSep->addChild(cc);
  cc->emissiveColor.setValue(1,1,1);
  cc->transparency.setValue(0.6);
  auto *ombvLogoTex=new SoTexture2;
  logoSep->addChild(ombvLogoTex);
  QIcon icon=Utils::QIconCached(":/openmbv.svg");
  QImage image=icon.pixmap(100, 100).toImage();
  int w=image.width();
  int h=image.height();
  // reorder image data
  vector<unsigned char> imageData(w*h*4);
  for(int y=0; y<h; ++y)
    for(int x=0; x<w; ++x) {
      QRgb pix=image.pixel(x, y);
      imageData[h*4*x+4*y+0]=qRed(pix);
      imageData[h*4*x+4*y+1]=qGreen(pix);
      imageData[h*4*x+4*y+2]=qBlue(pix);
      imageData[h*4*x+4*y+3]=qAlpha(pix);
    }
  // set inventor image
  ombvLogoTex->image.setValue(SbVec2s(w, h), 4, imageData.data());
  ombvLogoTex->wrapS.setValue(SoTexture2::CLAMP);
  ombvLogoTex->wrapT.setValue(SoTexture2::CLAMP);
  auto *ombvCoords=new SoCoordinate3;
  logoSep->addChild(ombvCoords);
  double size=0.15; // the logo filles maximal "size" of the screen
  ombvCoords->point.set1Value(0, 0, 0, 0);
  ombvCoords->point.set1Value(1, -size, 0, 0);
  ombvCoords->point.set1Value(2, -size, size, 0);
  ombvCoords->point.set1Value(3, 0, size, 0);
  auto *tc=new SoTextureCoordinate2;
  logoSep->addChild(tc);
  tc->point.set1Value(0, 1, 1);
  tc->point.set1Value(1, 1, 0);
  tc->point.set1Value(2, 0, 0);
  tc->point.set1Value(3, 0, 1);
  auto *ombvLogo=new SoFaceSet;
  logoSep->addChild(ombvLogo);
  ombvLogo->numVertices.set1Value(0, 4);
}

SoQtMyViewer::~SoQtMyViewer() {
  fgSep->unref();
  bgSep->unref();
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *const event) {
  // if D is down unset viewing
  if(event->isOfType(SoKeyboardEvent::getClassTypeId())) {
    auto *e=(SoKeyboardEvent*)event;
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

void SoQtMyViewer::actualRedraw() {
  // background
  getGLRenderAction()->apply(bgSep);

  // draw scene
  SoQtExaminerViewer::actualRedraw();

  // foreground (time and OpenMBV)
  glClear(GL_DEPTH_BUFFER_BIT);
  short x, y;
  getViewportRegion().getWindowSize().getValue(x, y);
  float ypos=font->size.getValue()+3;
  timeTrans->translation.setValue(-1+2.0/x*3, +1-2.0/y*ypos, 0);
  ombvTrans->translation.setValue(-1+2.0/x*3, -1+2.0/y*3, 0);
  ombvLogoTrans->translation.setValue(+1-2.0/x*3, -1+2.0/y*3, 0);
  ombvLogoScale->scaleFactor.setValue(x>y?(float)y/x:1,y>x?(float)x/y:1,1);
  getGLRenderAction()->apply(fgSep);

  // update fps
  MainWindow::getInstance()->fpsCB();
}

}
