#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/events/SoMouseButtonEvent.h>
#include <QApplication>
#include "mainwindow.h"

SoQtMyViewer::SoQtMyViewer(QWidget *parent) : SoQtExaminerViewer(parent) {
  setCameraType(SoOrthographicCamera::getClassTypeId());
  setDecoration(false);
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *const event) {
  if(((MainWindow*)(getParentWidget()->parentWidget()->parentWidget()))->soQtEventCB(event))
    return true;
  else
    return SoQtExaminerViewer::processSoEvent(event);
}
