#ifndef _SOQTMYVIEWER_H_
#define _SOQTMYVIEWER_H_

#include "config.h"
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <QEvent>

class SoQtMyViewer : public SoQtExaminerViewer {
  public:
    SoQtMyViewer(QWidget *parent);
  protected:
    SbBool processSoEvent(const SoEvent *const event);
};

#endif
