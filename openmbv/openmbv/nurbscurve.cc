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
#include "nurbscurve.h"
#include "utils.h"
#include <Inventor/nodes/SoNurbsCurve.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/nodes/SoCoordinate4.h>
#include "openmbvcppinterface/nurbscurve.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

NurbsCurve::NurbsCurve(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  nurbscurve=std::static_pointer_cast<OpenMBV::NurbsCurve>(obj);
  //iconFile="nurbscurve.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<vector<double> > cp = nurbscurve->getControlPoints();

  SoNode *points;
  if(cp[0].size()==3) {
    points = new SoCoordinate3;
    float pts[cp.size()][3];
    for(unsigned int i=0; i<cp.size(); i++) {
      for(unsigned int j=0; j<3; j++)
        pts[i][j] = cp[i][j];
    }
    static_cast<SoCoordinate3*>(points)->point.setValues(0, cp.size(), pts);
  }
  else {
    points = new SoCoordinate4;
    float pts[cp.size()][4];
    for(unsigned int i=0; i<cp.size(); i++) {
      for(unsigned int j=0; j<4; j++)
        pts[i][j] = cp[i][j];
    }
    static_cast<SoCoordinate4*>(points)->point.setValues(0, cp.size(), pts);
  }

  vector<double> knot = nurbscurve->getKnotVector();
  float u[knot.size()];
  for(unsigned int i=0; i<knot.size(); i++)
      u[i] = knot[i];

  auto *curve = new SoNurbsCurve;
  curve->numControlPoints.setValue(nurbscurve->getNumberOfControlPoints());
  curve->knotVector.setValues(0, knot.size(), u);
  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(curve);

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
}

void NurbsCurve::createProperties() {
  RigidBody::createProperties();
}

}
