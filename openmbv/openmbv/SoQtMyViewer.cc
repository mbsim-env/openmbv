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
#include "mytouchwidget.h"
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

SoQtMyViewer::SoQtMyViewer(QWidget *parent) : SoQtViewer(parent, nullptr, true, BROWSER, true) {
  static const char* OPENMBV_NO_MULTISAMPLING=getenv("OPENMBV_NO_MULTISAMPLING");
  if(!OPENMBV_NO_MULTISAMPLING)
    setSampleBuffers(4);
  updateTransperencySetting();
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

  // foreground (=screen annotation)
  screenAnnotationSep=new SoAnnotation;
  screenAnnotationSep->ref();

  // font size
  fontStyle=new SoFontStyle;
  fontStyle->family=SoFontStyle::SANS;
  screenAnnotationSep->addChild(fontStyle);

  screenAnnotationSep->addChild(MainWindow::getInstance()->getScreenAnnotationList());

  const float textHeight=0.03;

  // time (top left)
  {
    auto timeSep = new SoSeparator;
    screenAnnotationSep->addChild(timeSep);
    auto trans1 = new SoTranslation;
    trans1->translation.setValue(-1,1,0);
    timeSep->addChild(trans1);
    timeSep->addChild(MainWindow::getInstance()->getScreenAnnotationScale1To1());
    auto trans3 = new SoTranslation;
    trans3->translation.setValue(1,-1,0);
    timeSep->addChild(trans3);
    auto *soFgColorTop=new SoBaseColor;
    timeSep->addChild(soFgColorTop);
    soFgColorTop->rgb.connectFrom(MainWindow::getInstance()->getFgColorTop());
    auto trans = new SoTranslation;
    trans->translation.setValue(-0.99,1-textHeight,0);
    timeSep->addChild(trans);
    auto *fontStyle=new SoFontStyle;
    fontStyle->size.setValue(textHeight);
    fontStyle->family=SoFontStyle::TYPEWRITER;
    timeSep->addChild(fontStyle);
    timeSep->addChild(MainWindow::getInstance()->getTimeString());
  }

  {
    // ombvText (bottom right)
    auto ombvSep = new SoSeparator;
    screenAnnotationSep->addChild(ombvSep);
    auto trans1 = new SoTranslation;
    trans1->translation.setValue(1,-1,0);
    ombvSep->addChild(trans1);
    ombvSep->addChild(MainWindow::getInstance()->getScreenAnnotationScale1To1());
    auto trans3 = new SoTranslation;
    trans3->translation.setValue(-1,1,0);
    ombvSep->addChild(trans3);
    auto *soFgColorBottom=new SoBaseColor;
    ombvSep->addChild(soFgColorBottom);
    soFgColorBottom->rgb.connectFrom(MainWindow::getInstance()->getFgColorBottom());
    auto *textSep = new SoSeparator;
    ombvSep->addChild(textSep);
    auto trans = new SoTranslation;
    trans->translation.setValue(0.99,-0.99,0);
    textSep->addChild(trans);
    auto *fontStyle=new SoFontStyle;
    fontStyle->size.setValue(textHeight);
    fontStyle->family=SoFontStyle::SANS;
    textSep->addChild(fontStyle);
    auto *text2=new SoAsciiText;
    textSep->addChild(text2);
    text2->string.setValue("OpenMBV [https://www.mbsim-env.de/]");
    text2->justification.setValue(SoAsciiText::RIGHT);

    // ombvLogo (bottom right)
    auto *logoSep = new SoSeparator;
    ombvSep->addChild(logoSep);
    auto *cc=new SoMaterial;
    logoSep->addChild(cc);
    cc->emissiveColor.setValue(1,1,1);
    cc->transparency.setValue(0.6);
    auto *ombvLogoTex=new SoTexture2;
    logoSep->addChild(ombvLogoTex);
    QSvgRenderer svg(QString((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"icons"/"openmbv.svg").string().c_str()));
    QFontInfo fontinfo(parent->font());
    QImage image(fontinfo.pixelSize()*8, fontinfo.pixelSize()*8, QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&image);
    svg.render(&painter);
    // set inventor image
    ombvLogoTex->image.setValue(SbVec2s(image.width(), image.height()), 4, image.bits());
    ombvLogoTex->wrapS.setValue(SoTexture2::CLAMP);
    ombvLogoTex->wrapT.setValue(SoTexture2::CLAMP);
    auto *ombvCoords=new SoCoordinate3;
    logoSep->addChild(ombvCoords);
    double size=0.125; // the logo filles maximal "size" of the screen
    ombvCoords->point.set1Value(0, 1-size, -1+size+1.1*textHeight, 0); ombvCoords->point.set1Value(1, 1, -1+size+1.1*textHeight, 0);
    ombvCoords->point.set1Value(3, 1-size, -1+1.1*textHeight, 0);      ombvCoords->point.set1Value(2, 1, -1+1.1*textHeight, 0);
    auto *tc=new SoTextureCoordinate2;
    logoSep->addChild(tc);
    tc->point.set1Value(0, 0, 0); tc->point.set1Value(1, 1, 0);
    tc->point.set1Value(3, 0, 1); tc->point.set1Value(2, 1, 1);
    auto *ombvLogo=new SoFaceSet;
    logoSep->addChild(ombvLogo);
    ombvLogo->numVertices.set1Value(0, 4);
  }
}

SoQtMyViewer::~SoQtMyViewer() {
  screenAnnotationSep->unref();
  bgSep->unref();
}
 
void SoQtMyViewer::actualRedraw() {
  auto mw = MainWindow::getInstance();
  short x, y;
  getViewportRegion().getWindowSize().getValue(x, y);
  if(getCamera()->getStereoMode()!=SoCamera::MONOSCOPIC)
    getCamera()->aspectRatio.setValue(static_cast<float>(x)/y*aspectRatio);
  else
    getCamera()->aspectRatio.setValue(1.0);

  if(mw->getBackgroundNeeded()) {
    glClear(GL_DEPTH_BUFFER_BIT);
    // background
    getGLRenderAction()->apply(bgSep);
  }

  // draw scene
  SoQtViewer::actualRedraw();

  // foreground (time and OpenMBV)
  auto ombvLogoScaleX=x>y?(float)y/x:1;
  auto ombvLogoScaleY=y>x?(float)x/y:1;
  if(aspectRatio>1)
    mw->getScreenAnnotationScale1To1()->scaleFactor.setValue(ombvLogoScaleX/aspectRatio,ombvLogoScaleY,1);
  else
    mw->getScreenAnnotationScale1To1()->scaleFactor.setValue(ombvLogoScaleX,ombvLogoScaleY*aspectRatio,1);
  getGLRenderAction()->apply(screenAnnotationSep);

  // update fps
  mw->fpsCB();
}

void SoQtMyViewer::setAspectRatio(double r) {
  aspectRatio=r;
}

void SoQtMyViewer::updateTransperencySetting() {
  switch(appSettings->get<int>(AppSettings::transparency)) {
    case 0:
      setTransparencyType(SoGLRenderAction::BLEND);
      break;
    case 1:
      setTransparencyType(SoGLRenderAction::DELAYED_BLEND);
      break;
    case 2:
      setTransparencyType(SoGLRenderAction::SORTED_OBJECT_BLEND);
      break;
    case 3:
      setTransparencyType(SoGLRenderAction::SORTED_OBJECT_SORTED_TRIANGLE_BLEND);
      break;
    case 4:
      setAlphaChannel(true);
      setTransparencyType(SoGLRenderAction::SORTED_LAYERS_BLEND);
      break;
    case 5:
      setTransparencyType(SoGLRenderAction::NONE);
      break;
  }
}

void SoQtMyViewer::processEvent(QEvent *event) {
  auto type = event->type();
  if(type==QEvent::KeyPress || type==QEvent::KeyRelease) {
    auto keyEvent = static_cast<QKeyEvent*>(event);
    if(keyEvent->key()==Qt::Key_D) {
      if(keyEvent->isAutoRepeat()) // do not propagate auto repeat keys for the dragger key to coin3d
        return;
      draggerKeyDown = type==QEvent::KeyPress; // save the the dragger key is currently down (coin3d has only wasShiftDown/...)
    }
  }
  SoQtViewer::processEvent(event);
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *event) {
  // if the last call has set draggerActiveReset then we reset draggerActive to draggerActiveLast before continouing
  if(draggerActiveReset) {
    draggerActive = draggerActiveLast;
    draggerActiveReset = false;
  }

  // update dragger active state
  if(event->getTypeId()==SoMouseButtonEvent::getClassTypeId()) {
    // the dragger state keeps the same from the beginning of a mouse button 1-3 down event
    // up to the corresponding mouse button 1-3 up event
    auto mouseEvent = static_cast<const SoMouseButtonEvent*>(event);
    if(mouseEvent->getButton()==SoMouseButtonEvent::BUTTON1 ||
       mouseEvent->getButton()==SoMouseButtonEvent::BUTTON2 ||
       mouseEvent->getButton()==SoMouseButtonEvent::BUTTON3) {
      if(mouseEvent->getState()==SoMouseButtonEvent::DOWN) {
        // on mouse button down -> save current draggerActive and set new draggerActive according the modifier state
        draggerActiveLast = draggerActive;
        draggerActive = draggerKeyDown;
      }
      else if(mouseEvent->getState()==SoMouseButtonEvent::UP)
        // on mouse button up -> set the draggerActiveReset flag to reset the dragger on the next call
        draggerActiveReset = true;
    }
    // for a mouse button 4,5 (wheel) only the down event is emitted (no start end range for dragger state)
    if(mouseEvent->getButton()==SoMouseButtonEvent::BUTTON4 ||
       mouseEvent->getButton()==SoMouseButtonEvent::BUTTON5) {
      draggerActiveLast = draggerActive;
      draggerActive = draggerKeyDown;
      draggerActiveReset = true;
    }
  }
  // touch events are not propagated to SoQt (for now) -> so we cannot handle dragger with touch for now

  // disable/enable 3D cursor during dragger movement
  if(appSettings->get<bool>(AppSettings::mouseCursor3D))
    MainWindow::getInstance()->glViewerWG->setCursor3D(!draggerActive);

  // pass Ctrl key events to coin since Ctrl may be used by draggers
  if(event->getTypeId()==SoKeyboardEvent::getClassTypeId()) {
    auto keyEvent = static_cast<const SoKeyboardEvent*>(event);
    if(keyEvent->getKey()==SoKeyboardEvent::LEFT_CONTROL || keyEvent->getKey()==SoKeyboardEvent::RIGHT_CONTROL)
      return SoQtViewer::processSoEvent(event);
  }

  if(draggerActive)
    // if dragger is active -> pass event to the scene graph
    return SoQtViewer::processSoEvent(event);
  else
    // if dragger is not active -> do not pass the event to the scene graph
    return true;
}

}
