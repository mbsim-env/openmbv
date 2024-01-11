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
#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoAnnotation.h>
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
#include <Inventor/nodes/SoAsciiText.h>
#include <QApplication>
#include <QDesktopWidget>
#include <QSvgRenderer>
#include <QPainter>
#include "mainwindow.h"
#include <boost/dll.hpp>

using namespace std;

namespace OpenMBVGUI {

constexpr float fontScale = 0.0025;

SoQtMyViewer::SoQtMyViewer(QWidget *parent, int transparency) : SoQtViewer(parent, nullptr, true, BROWSER, true) {
  static const char* OPENMBV_NO_MULTISAMPLING=getenv("OPENMBV_NO_MULTISAMPLING");
  if(!OPENMBV_NO_MULTISAMPLING)
    setSampleBuffers(4);
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
  setSeekTime(1);

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
  fgSep=new SoAnnotation;
  fgSep->ref();
  // font size
  font=new SoFont;
  font->size.setValue(15);
#ifdef _WIN32
  font->name.setValue("Lucida Sans Typewriter");
#else
  font->name.setValue("DejaVu Sans Mono");
#endif
  fgSep->addChild(font);
  // time (top left)
  auto *timeSep=new SoSeparator;
  fgSep->addChild(timeSep);
  timeTrans=new SoTranslation;
  timeSep->addChild(timeTrans);
  auto *soFgColorTop=new SoBaseColor;
  text2Scale=new SoScale;
  text2Scale->scaleFactor.setValue(fontScale,fontScale,fontScale);
  ombvLogoScale=new SoScale;
  timeSep->addChild(text2Scale);
  timeSep->addChild(ombvLogoScale);
  timeSep->addChild(soFgColorTop);
  soFgColorTop->rgb.connectFrom(MainWindow::getInstance()->getFgColorTop());
  timeSep->addChild(MainWindow::getInstance()->getTimeString());
  // ombvText (bottom left)
  auto *ombvSep=new SoSeparator;
  fgSep->addChild(ombvSep);
  ombvTrans=new SoTranslation;
  ombvSep->addChild(ombvTrans);
  auto *text2=new SoAsciiText;
  auto *soFgColorBottom=new SoBaseColor;
  ombvSep->addChild(soFgColorBottom);
  soFgColorBottom->rgb.connectFrom(MainWindow::getInstance()->getFgColorBottom());
  ombvSep->addChild(text2Scale);
  ombvSep->addChild(ombvLogoScale);
  ombvSep->addChild(text2);
  text2->string.setValue("OpenMBV [https://www.mbsim-env.de/]");
  // ombvLogo (bottom right)
  auto *logoSep=new SoSeparator;
  fgSep->addChild(logoSep);
  ombvLogoTrans=new SoTranslation;
  logoSep->addChild(ombvLogoTrans);
  logoSep->addChild(ombvLogoScale);
  auto *cc=new SoMaterial;
  logoSep->addChild(cc);
  cc->emissiveColor.setValue(1,1,1);
  cc->transparency.setValue(0.6);
  auto *ombvLogoTex=new SoTexture2;
  logoSep->addChild(ombvLogoTex);
  QSvgRenderer svg(QString((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"icons"/"openmbv.svg").string().c_str()));
  QFontInfo fontinfo(parent->font());
  QImage image(fontinfo.pixelSize()*5, fontinfo.pixelSize()*5, QImage::Format_RGBA8888);
  QPainter painter(&image);
  svg.render(&painter);
  // set inventor image
  ombvLogoTex->image.setValue(SbVec2s(image.width(), image.height()), 4, image.bits());
  ombvLogoTex->wrapS.setValue(SoTexture2::CLAMP);
  ombvLogoTex->wrapT.setValue(SoTexture2::CLAMP);
  auto *ombvCoords=new SoCoordinate3;
  logoSep->addChild(ombvCoords);
  double size=0.15; // the logo filles maximal "size" of the screen
  ombvCoords->point.set1Value(0, -size, 0, 0);    ombvCoords->point.set1Value(1, 0, 0, 0);
  ombvCoords->point.set1Value(3, -size, size, 0); ombvCoords->point.set1Value(2, 0, size, 0);
  auto *tc=new SoTextureCoordinate2;
  logoSep->addChild(tc);
  tc->point.set1Value(0, 0, 1); tc->point.set1Value(1, 1, 1);
  tc->point.set1Value(3, 0, 0); tc->point.set1Value(2, 1, 0);
  auto *ombvLogo=new SoFaceSet;
  logoSep->addChild(ombvLogo);
  ombvLogo->numVertices.set1Value(0, 4);
}

SoQtMyViewer::~SoQtMyViewer() {
  fgSep->unref();
  bgSep->unref();
}
 
void SoQtMyViewer::actualRedraw() {
  static bool nearPlaneByDistance=getenv("OPENMBV_NEARPLANEBYDISTANCE")!=nullptr;
  if(nearPlaneByDistance)
    setAutoClippingStrategy(CONSTANT_NEAR_PLANE, MainWindow::getInstance()->nearPlaneValue);
  else
    setAutoClippingStrategy(VARIABLE_NEAR_PLANE, MainWindow::getInstance()->nearPlaneValue);

  short x, y;
  getViewportRegion().getWindowSize().getValue(x, y);
  if(getCamera()->getStereoMode()!=SoCamera::MONOSCOPIC)
    getCamera()->aspectRatio.setValue(static_cast<float>(x)/y*aspectRatio);
  else
    getCamera()->aspectRatio.setValue(1.0);

  glClear(GL_DEPTH_BUFFER_BIT);
  // background
  getGLRenderAction()->apply(bgSep);

  // draw scene
  SoQtViewer::actualRedraw();

  // foreground (time and OpenMBV)
  float ypos=400*font->size.getValue()*text2Scale->scaleFactor.getValue().getValue()[1]*ombvLogoScale->scaleFactor.getValue().getValue()[1];
  timeTrans->translation.setValue(-1+2.0/x*3, +1-2.0/y*ypos, 0);
  ombvTrans->translation.setValue(-1+2.0/x*3, -1+2.0/y*3, 0);
  ombvLogoTrans->translation.setValue(+1-2.0/x*3, -1+2.0/y*3, 0);
  auto ombvLogoScaleX=x>y?(float)y/x:1;
  auto ombvLogoScaleY=y>x?(float)x/y:1;
  if(aspectRatio>1)
    ombvLogoScale->scaleFactor.setValue(ombvLogoScaleX/aspectRatio,ombvLogoScaleY,1);
  else
    ombvLogoScale->scaleFactor.setValue(ombvLogoScaleX,ombvLogoScaleY*aspectRatio,1);
  getGLRenderAction()->apply(fgSep);

  // update fps
  MainWindow::getInstance()->fpsCB();
}

void SoQtMyViewer::setAspectRatio(double r) {
  aspectRatio=r;
}

}
