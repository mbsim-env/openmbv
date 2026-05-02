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

#ifndef _OPENMBVGUI_ARROW_H_
#define _OPENMBVGUI_ARROW_H_

#include "dynamiccoloredbody.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoRotation.h>
#include <QMenu>
#include <hdf5serie/vectorserie.h>
#include "editors.h"
#include "openmbvcppinterface/arrow.h"

namespace OpenMBV {
  class Arrow;
}

namespace OpenMBVGUI {

class Arrow : public DynamicColoredBody {
  Q_OBJECT
  protected:
    std::shared_ptr<OpenMBV::Arrow> arrow;
    SoSwitch *soPathSwitch;
    std::array<SoSwitch*,3> soArrowSwitch { nullptr, nullptr, nullptr };
    std::array<SoSwitch*,3> soOutLineSwitch { nullptr, nullptr, nullptr };
    std::array<SoSeparator*,3> soOutLineSep { nullptr, nullptr, nullptr };
    SoCoordinate3 *pathCoord;
    std::array<SoCoordinate3*,3> lineCoord { nullptr, nullptr, nullptr };
    SoLineSet *pathLine;
    std::array<SoTranslation*,3> toPoint, bTrans;
    std::array<SoRotation*,3> rotation1, rotation2;
    int pathMaxFrameRead;
    bool pathNewLine;
    double update() override;
    double updateComponents(int c);
    std::array<SoScale*,3> scale1, scale2;
    double headLength;
    OpenMBV::Arrow::Components components;
    std::array<std::vector<OpenMBV::Float>,3> data;
    std::array<double,3> length;
    double scaleLength;
    void createProperties() override;
    void ctorComponents(int c);
  public:
    Arrow(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    QString getInfo() override;
};

}

#endif
