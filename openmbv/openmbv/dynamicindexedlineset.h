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

#ifndef _OPENMBVGUI_DYNAMICINDEXEDLINESET_H_
#define _OPENMBVGUI_DYNAMICINDEXEDLINESET_H_

#include "flexiblebody.h"
#include <string>
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoMaterial.h>

namespace OpenMBV {
  class DynamicIndexedLineSet;
}

namespace OpenMBVGUI {

class DynamicIndexedLineSet : public FlexibleBody {
  Q_OBJECT
  public:
    DynamicIndexedLineSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
  protected:
    std::shared_ptr<OpenMBV::DynamicIndexedLineSet> lineset;
//    double update() override;
};

}

#endif
