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

#ifndef _MBSIMGUI_ARROW_H_
#define _MBSIMGUI_ARROW_H_

#include "dynamiccoloredbody.h"
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoRotation.h>
#include <QtGui/QMenu>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>
#include "editors.h"

namespace OpenMBV {
  class Arrow;
}

namespace OpenMBVGUI {

class Arrow : public DynamicColoredBody {
  Q_OBJECT
  protected:
    OpenMBV::Arrow *arrow;
    SoSwitch *soPathSwitch;
    SoCoordinate3 *pathCoord, *lineCoord;
    SoLineSet *pathLine;
    SoMaterial *mat;
    SoBaseColor *baseColor;
    SoTranslation *toPoint, *bTrans;
    SoRotation *rotation1, *rotation2;
    int pathMaxFrameRead;
    virtual double update();
    SoScale *scale1, *scale2;
    double headLength;
    std::vector<double> data;
    double length, scaleLength;
  public:
    Arrow(OpenMBV::Object* obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    virtual QString getInfo();
};

}

#endif
