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

#ifndef _OPENMBVGUI_SOQTMYVIEWER_H_
#define _OPENMBVGUI_SOQTMYVIEWER_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/Qt/viewers/SoQtViewer.h>
#include <QtCore/QEvent>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN // GL/gl.h includes windows.h on Windows -> avoid full header -> WIN32_LEAN_AND_MEAN
#endif
#include <GL/gl.h>
#include <Inventor/fields/SoMFColor.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoScale.h>

namespace OpenMBVGUI {

class SoQtMyViewer : public SoQtViewer {
  public:
    SoQtMyViewer(QWidget *parent, int transparency);
    ~SoQtMyViewer() override;
    void setAspectRatio(double r);
    void changeCameraValues(SoCamera *cam) override { SoQtViewer::changeCameraValues(cam); } // is protected
  protected:
    SbBool processSoEvent(const SoEvent *event) override { return true; } // disable So events
    void actualRedraw() override;

    // for text in viewport

    SoSeparator *fgSep, *bgSep;
    SoTranslation *timeTrans, *ombvTrans, *ombvLogoTrans;
    float aspectRatio { 1.0 };
    SoScale *ombvLogoScale;
    SoFont *font;
    SoScale *text2Scale;

    friend class MainWindow;
};

}

#endif
