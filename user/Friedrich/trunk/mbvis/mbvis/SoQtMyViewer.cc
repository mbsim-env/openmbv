#include "SoQtMyViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>

SoQtMyViewer::SoQtMyViewer(QWidget *parent) : SoQtExaminerViewer(parent) {
  setCameraType(SoOrthographicCamera::getClassTypeId());
  setDecoration(false);
}

SbBool SoQtMyViewer::processSoEvent(const SoEvent *const ev) {
  return SoQtExaminerViewer::processSoEvent(ev);
}
