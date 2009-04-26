#ifndef _SOQTMYVIEWER_H_
#define _SOQTMYVIEWER_H_

#include "config.h"
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <QEvent>
#include <GL/gl.h>
#include <Inventor/nodes/SoText2.h>

class SoQtMyViewer : public SoQtExaminerViewer {
  public:
    SoQtMyViewer(QWidget *parent, SoText2* timeString_);
    void setSeekMode(SbBool enabled) { SoQtExaminerViewer::setSeekMode(enabled); } // is protected
    void seekToPoint(const SbVec3f& scenepos) { SoQtExaminerViewer::seekToPoint(scenepos); } // is protected
  protected:
    SbBool processSoEvent(const SoEvent *const event);
    virtual void actualRedraw(void);
    // for text in viewport
    SoSeparator *fgSep;
    SoTranslation *timeTrans, *ombvTrans;
};

#endif
