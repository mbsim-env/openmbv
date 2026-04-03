/* OpenMBV - Open Multi Body Viewer.
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
#include "rigidspineextrusion.h"
#include "openmbvcppinterface/rigidspineextrusion.h"

using namespace std;

namespace OpenMBVGUI {

RigidSpineExtrusion::RigidSpineExtrusion(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  rse = static_pointer_cast<OpenMBV::RigidSpineExtrusion>(obj);

  extrusion.init(rse->getSpine().size(), rse->getContour(), 1.0, rse->getCounterClockWise(), soSepRigidBody, soOutLineSep);
  vector<OpenMBV::Float> data;
  data.reserve(1+6*rse->getSpine().size());
  data.emplace_back(0.0); // the first data is the time -> 0 for a rigid body
  for(const auto& s : rse->getSpine()) {
    data.emplace_back(s.x);
    data.emplace_back(s.y);
    data.emplace_back(s.z);
    data.emplace_back(s.alpha);
    data.emplace_back(s.beta);
    data.emplace_back(s.gamma);
  }
  extrusion.setCardanWrtWorldSpine(data);

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
}

}
