#ifndef _SOQTMYVIEWER_H_
#define _SOQTMYVIEWER_H_

#include "config.h"
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <QEvent>
#include <GL/gl.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSeparator.h>

class SoQtMyViewer : public SoQtExaminerViewer {
  public:
    SoQtMyViewer(QWidget *parent);
  protected:
    SbBool processSoEvent(const SoEvent *const event);
    void actualRedraw(void) {
      SoQtExaminerViewer::actualRedraw();

      glClear(GL_DEPTH_BUFFER_BIT);
      short x, y;
      getViewportRegion().getWindowSize().getValue(x, y);
      SoSeparator *fg=new SoSeparator;
      fg->ref();
      SoTranslation *t=new SoTranslation;
      fg->addChild(t);
      t->translation.setValue(-1+2.0/x*3,1-2.0/y*15,0);
      SoText2 *text=new SoText2;
      fg->addChild(text);
      text->string.setValue("sdflk");
      SoTranslation *t2=new SoTranslation;
      fg->addChild(t2);
      t2->translation.setValue(0,-1+2.0/y*15 -1+2.0/y*3,0);
      SoText2 *text2=new SoText2;
      fg->addChild(text2);
      text2->string.setValue("OpenMBV [http://openmbv.berlios.de]");
      getGLRenderAction()->apply(fg);
    }

};

#endif
