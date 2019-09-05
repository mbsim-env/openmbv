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

#include "config.h"
#include "planargear.h"
#include <Inventor/nodes/SoCylinder.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/planargear.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

PlanarGear::PlanarGear(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::PlanarGear>(obj);
  iconFile="planargear.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int nz = e->getNumberOfTeeth();
  double h = e->getHeight();
  double be = e->getHelixAngle();
  double al0 = e->getPressureAngle();
  double m = e->getModule();
  double b = e->getBacklash();

  double d0 = m*nz/cos(be);
  double r0 = d0/2;
  int numb = 10;
  double du = 0.3/numb;
  vector<double> x(numb+1), y(numb+1), z(numb+1);

  auto *cyl = new SoCylinder;
  soSepRigidBody->addChild(cyl);
  cyl->radius.setValue(m*nz/2);
  cyl->height.setValue(h);

  for(int j=0; j<2; j++) {
    int signj=j?-1:1;
    double u = -0.15;
    for (int i=0; i<=numb; i++) {
      x[i] = -signj*u*sin(al0)*cos(be);
      y[i] = u*cos(al0);
      z[i] = -signj*u*sin(al0)*sin(be)+r0;
      u += du;
    }
    float pts[x.size()][3];
    for(unsigned int i=0; i<x.size(); i++) {
      pts[i][0] = x[i];
      pts[i][1] = y[i];
      pts[i][2] = z[i];
    }
    auto *r = new SoRotation;
    soSepRigidBody->addChild(r);
    r->rotation.setValue(SbVec3f(0,1,0),(j?-2:1)*(M_PI/2-b/m)/nz);
    for(int k=0; k<nz; k++) {
      auto *points = new SoCoordinate3;
      auto *line = new SoLineSet;
      points->point.setValues(0, x.size(), pts);
      line->numVertices.setValue(x.size());
      soSepRigidBody->addChild(points);
      soSepRigidBody->addChild(line);
      auto *r = new SoRotation;
      soSepRigidBody->addChild(r);
      r->rotation.setValue(SbVec3f(0,1,0),signj*2*M_PI/nz);
    }
  }
}
 
void PlanarGear::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(5, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getNumberOfTeeth, &OpenMBV::PlanarGear::setNumberOfTeeth);
    FloatEditor *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getHeight, &OpenMBV::Rack::setHeight);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getWidth, &OpenMBV::PlanarGear::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getHelixAngle, &OpenMBV::PlanarGear::setHelixAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getModule, &OpenMBV::PlanarGear::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getPressureAngle, &OpenMBV::PlanarGear::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getBacklash, &OpenMBV::PlanarGear::setBacklash);
  }
}

}
