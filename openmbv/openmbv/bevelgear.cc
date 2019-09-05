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
#include "bevelgear.h"
#include <Inventor/nodes/SoCylinder.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/bevelgear.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

BevelGear::BevelGear(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::BevelGear>(obj);
  iconFile="bevelgear.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int nz = e->getNumberOfTeeth();
  double width = e->getWidth();
  double be = e->getHelixAngle();
  double ga = e->getPitchAngle();
  double al0 = e->getPressureAngle();
  double m = e->getModule();
  double b = e->getBacklash();

  double d0 = m*nz/cos(be);
  double r0 = d0/2;
  int numb = 5;

  double R = r0/sin(ga);
  double d = sqrt(R*R-r0*r0);
  double dphi = 60./180*M_PI/2/numb;
  vector<double> x(numb+1), y(numb+1), z(numb+1);
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;
    double phi = -30./180*M_PI/2;
    for (int i=0; i<=numb; i++) {
      double phi2 = -phi*R/r0;
      double u = -signj*sin(phi)/cos(phi-be)*sin(al0)*R;
      double a = signj*u*sin(al0)*cos(phi-be)+R*sin(phi);
      double b = u*cos(al0)-d*sin(ga);
      double c = -signj*u*sin(al0)*sin(phi-be)+R*cos(phi)-d*cos(ga);
      x[i] = -a*cos(phi2)+b*sin(phi2)*cos(ga)-c*sin(phi2)*sin(ga);
      y[i] = a*sin(phi2)+b*cos(phi2)*cos(ga)-c*cos(phi2)*sin(ga);
      z[i] = b*sin(ga)+c*cos(ga);
      phi += dphi;
    }
    float pts[x.size()][3];
    for(unsigned int i=0; i<x.size(); i++) {
      pts[i][0] = x[i];
      pts[i][1] = y[i];
      pts[i][2] = z[i];
    }
    auto *r = new SoRotation;
    soSepRigidBody->addChild(r);
    r->rotation.setValue(SbVec3f(0,0,1),(j?-2:1)*(-M_PI/2+b/m)/nz);
    for(int k=0; k<nz; k++) {
      auto *points = new SoCoordinate3;
      auto *line = new SoLineSet;
      points->point.setValues(0, x.size(), pts);
      line->numVertices.setValue(x.size());
      soSepRigidBody->addChild(points);
      soSepRigidBody->addChild(line);
      auto *r = new SoRotation;
      soSepRigidBody->addChild(r);
      r->rotation.setValue(SbVec3f(0,0,1),signj*2*M_PI/nz);
    }
  }

  auto *r = new SoRotation;
  soSepRigidBody->addChild(r);
  r->rotation.setValue(SbVec3f(1,0,0),M_PI/2);

  auto *cyl = new SoCylinder;
  soSepRigidBody->addChild(cyl);
  cyl->radius.setValue(m*nz/2);
  cyl->height.setValue(width);
}
 
void BevelGear::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(5, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getNumberOfTeeth, &OpenMBV::BevelGear::setNumberOfTeeth);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getWidth, &OpenMBV::BevelGear::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getHelixAngle, &OpenMBV::BevelGear::setHelixAngle);
    FloatEditor *pitchAngleEditor=new FloatEditor(properties, QIcon(), "Pitch angle");
    pitchAngleEditor->setRange(-M_PI/4, M_PI/4);
    pitchAngleEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getPitchAngle, &OpenMBV::BevelGear::setPitchAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getModule, &OpenMBV::BevelGear::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getPressureAngle, &OpenMBV::BevelGear::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::BevelGear::getBacklash, &OpenMBV::BevelGear::setBacklash);
  }
}

}
