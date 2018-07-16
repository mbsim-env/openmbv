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

#ifndef _OPENMBVGUI_DYNAMICCOLOREDBODY_H_
#define _OPENMBVGUI_DYNAMICCOLOREDBODY_H_

#include "body.h"
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoBaseColor.h>
#include "editors.h"

namespace OpenMBV {
  class DynamicColoredBody;
}

namespace OpenMBVGUI {

class DynamicColoredBody : public Body {
  Q_OBJECT
  protected:
    double minimalColorValue, maximalColorValue;
    SoMaterial *mat;
    SoBaseColor *baseColor;
    std::vector<double> diffuseColor;
    double color,oldColor;
    void setColor(double col);
    void setHueColor(double hue);
    double getColor() { return color; }
    std::shared_ptr<OpenMBV::DynamicColoredBody> dcb;
    void createProperties() override;
  public:
    DynamicColoredBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind, bool perVertexIndexed=false);
    QString getInfo() override;
};

}

#endif
