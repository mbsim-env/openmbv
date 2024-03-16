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

#ifndef _OPENMBVGUI_IVSCREENANNOTATION_H_
#define _OPENMBVGUI_IVSCREENANNOTATION_H_

#include <openmbv/body.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>

namespace OpenMBV {
  class IvScreenAnnotation;
}

class SoAlphaTest;
class SoLineSet;

namespace OpenMBVGUI {

class IvScreenAnnotation : public Body {
  Q_OBJECT
  public:
    IvScreenAnnotation(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~IvScreenAnnotation() override;
    double update() override;
  protected:
    std::shared_ptr<OpenMBV::IvScreenAnnotation> ivsa;
    std::vector<SoAlphaTest*> columnLabelFields;
    SoSeparator *sep;

    SoCoordinate3 *pathCoord;
    SoLineSet *pathLine;
    int pathMaxFrameRead;
    SoPath *pathPath { nullptr };
    std::unique_ptr<SoSearchAction> sa;
    std::unique_ptr<SoGetMatrixAction> gma;
};

}

#endif
