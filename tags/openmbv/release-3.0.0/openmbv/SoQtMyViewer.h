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

#ifndef _SOQTMYVIEWER_H_
#define _SOQTMYVIEWER_H_

#include "config.h"
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <QEvent>
#include <GL/gl.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/fields/SoMFColor.h>

class SoQtMyViewer : public SoQtExaminerViewer {
  public:
    SoQtMyViewer(QWidget *parent);
    void setSeekMode(SbBool enabled) { SoQtExaminerViewer::setSeekMode(enabled); } // is protected
    void seekToPoint(const SbVec3f& scenepos) { SoQtExaminerViewer::seekToPoint(scenepos); } // is protected
    void myChangeCameraValues(SoCamera *cam) { changeCameraValues(cam); } // is protected
  protected:
    SbBool processSoEvent(const SoEvent *const event);
    virtual void actualRedraw(void);
    // for text in viewport
  public:
    SoSeparator *fgSep, *bgSep;
    SoTranslation *timeTrans, *ombvTrans;
};

#endif
