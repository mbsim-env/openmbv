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
#include "nurbssurface.h"
#include "utils.h"
#include <Inventor/nodes/SoNurbsSurface.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/nodes/SoCoordinate4.h>
#include "openmbvcppinterface/nurbssurface.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

NurbsSurface::NurbsSurface(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  nurbssurface=std::static_pointer_cast<OpenMBV::NurbsSurface>(obj);
  iconFile="nurbssurface.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<vector<double> > cp = nurbssurface->getControlPoints();

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

  vector<double> uKnot = nurbssurface->getUKnotVector();
  float u[uKnot.size()];
  for(unsigned int i=0; i<uKnot.size(); i++)
      u[i] = uKnot[i];

  vector<double> vKnot = nurbssurface->getVKnotVector();
  float v[vKnot.size()];
  for(unsigned int i=0; i<vKnot.size(); i++)
      v[i] = vKnot[i];

  auto *surface = new SoNurbsSurface;
  surface->numUControlPoints.setValue(nurbssurface->getNumberOfUControlPoints());
  surface->numVControlPoints.setValue(nurbssurface->getNumberOfVControlPoints());
  surface->uKnotVector.setValues(0, uKnot.size(), u);
  surface->vKnotVector.setValues(0, vKnot.size(), v);
  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(surface);

  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
}

void NurbsSurface::createProperties() {
  RigidBody::createProperties();
}

}
