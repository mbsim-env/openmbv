#include "config.h"
#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <QApplication>
#include "mainwindow.h"

SoQtMyViewer::SoQtMyViewer(QWidget *parent, SoText2 *timeString_) : SoQtExaminerViewer(parent) {
  setCameraType(SoOrthographicCamera::getClassTypeId());
  setDecoration(false);
  timeString=timeString_;
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
  SoSeparator *fg=new SoSeparator;
  fg->ref();
  SoTranslation *t=new SoTranslation;
  fg->addChild(t);
  t->translation.setValue(-1+2.0/x*3,1-2.0/y*15,0);
  fg->addChild(timeString);
  SoTranslation *t2=new SoTranslation;
  fg->addChild(t2);
  t2->translation.setValue(0,-1+2.0/y*15 -1+2.0/y*3,0);
  SoText2 *text2=new SoText2;
  fg->addChild(text2);
  text2->string.setValue("OpenMBV [http://openmbv.berlios.de]");
  getGLRenderAction()->apply(fg);
}
