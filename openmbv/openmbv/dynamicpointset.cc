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

#include "config.h"
#include "dynamicpointset.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/dynamicpointset.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

DynamicPointSet::DynamicPointSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : FlexibleBody(obj, parentItem, soParent, ind) {
  pointset=std::static_pointer_cast<OpenMBV::DynamicPointSet>(obj);
  iconFile="pointset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  auto *point = new SoPointSet;
  soSep->addChild(point);
}

}
