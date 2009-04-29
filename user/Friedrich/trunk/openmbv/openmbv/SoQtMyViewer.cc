#include "config.h"
#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <QApplication>
#include "mainwindow.h"

SoQtMyViewer::SoQtMyViewer(QWidget *parent, SoText2 *timeString) : SoQtExaminerViewer(parent) {
  setCameraType(SoOrthographicCamera::getClassTypeId());
  setDecoration(false);
  setTransparencyType(SoGLRenderAction::SORTED_OBJECT_BLEND);
  setAnimationEnabled(false);
  setSeekTime(1);

  fgSep=new SoSeparator;
  fgSep->ref();
  timeTrans=new SoTranslation;
  fgSep->addChild(timeTrans);
  fgSep->addChild(timeString);
  ombvTrans=new SoTranslation;
  fgSep->addChild(ombvTrans);
  SoText2 *text2=new SoText2;
  fgSep->addChild(text2);
  text2->string.setValue("OpenMBV [http://openmbv.berlios.de]");
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *const event) {
  if(MainWindow::getInstance()->soQtEventCB(event))
    return true;
  else
    return SoQtExaminerViewer::processSoEvent(event);
}

void SoQtMyViewer::actualRedraw(void) {
  // draw scene
  SoQtExaminerViewer::actualRedraw();

  // overlay time an OpenMBV
  glClear(GL_DEPTH_BUFFER_BIT);
  short x, y;
  getViewportRegion().getWindowSize().getValue(x, y);
  timeTrans->translation.setValue(-1+2.0/x*3,1-2.0/y*15,0);
  ombvTrans->translation.setValue(0,-1+2.0/y*15 -1+2.0/y*3,0);
  getGLRenderAction()->apply(fgSep);

  // update fps
  MainWindow::getInstance()->fpsCB();
}
