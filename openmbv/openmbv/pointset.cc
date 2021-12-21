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
#include "pointset.h"
#include "utils.h"
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/pointset.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

PointSet::PointSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  pointset=std::static_pointer_cast<OpenMBV::PointSet>(obj);
  iconFile="pointset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<vector<double> > vp = pointset->getVertexPositions();

  float pts[vp.size()][3];
  for(unsigned int i=0; i<vp.size(); i++) {
    for(unsigned int j=0; j<3; j++)
      pts[i][j] = vp[i][j];
  }

  auto *points = new SoCoordinate3;
  auto *pointset = new SoPointSet;
  points->point.setValues(0, vp.size(), pts);
  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(pointset);
}

void PointSet::createProperties() {
  RigidBody::createProperties();
}

}
