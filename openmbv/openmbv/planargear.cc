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
  double be = e->getHelixAngle();
  double al0 = e->getPressureAngle();
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();
  double w = e->getWidth();
  double h = e->getHeight();

  double d0 = m*nz;
  double r0 = d0/2;
  double dphi = (M_PI/2-b/m)/nz;

  vector<double> x(8), y(8), z(8);
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;
    double eta = -e->getModule()/cos(al0);
    double s = w/2;
    double xi = (s-eta*sin(al0)*sin(be))/cos(be);
    x[4*j+0] = -eta*sin(al0)*cos(be)+xi*sin(be);
    y[4*j+0] = signj*eta*cos(al0);
    z[4*j+0] = eta*sin(al0)*sin(be)+r0+xi*cos(be);
    s = -w/2;
    xi = (s-eta*sin(al0)*sin(be))/cos(be);
    x[4*j+1] = -eta*sin(al0)*cos(be)+xi*sin(be);
    y[4*j+1] = signj*eta*cos(al0);
    z[4*j+1] = eta*sin(al0)*sin(be)+r0+xi*cos(be);
    eta = e->getModule()/cos(al0);
    s = -w/2;
    xi = (s-eta*sin(al0)*sin(be))/cos(be);
    x[4*j+2] = -eta*sin(al0)*cos(be)+xi*sin(be);
    y[4*j+2] = signj*eta*cos(al0);
    z[4*j+2] = eta*sin(al0)*sin(be)+r0+xi*cos(be);
    s = w/2;
    xi = (s-eta*sin(al0)*sin(be))/cos(be);
    x[4*j+3] = -eta*sin(al0)*cos(be)+xi*sin(be);
    y[4*j+3] = signj*eta*cos(al0);
    z[4*j+3] = eta*sin(al0)*sin(be)+r0+xi*cos(be);
    for(int i=4*j; i<4*j+4; i++) {
      x[i] = cos(dphi)*x[i] + signj*sin(dphi)*z[i];
      z[i] = -signj*sin(dphi)*x[i] + cos(dphi)*z[i];
    }
  }

  int indices[25];
  for(int i=0; i<4; i++)
    indices[i] = i;
  indices[4] = -1;
  for(int i=5; i<9; i++)
    indices[i] = i-1;
  indices[9] = -1;
  // top
  indices[10] = 5;
  indices[11] = 4;
  indices[12] = 3;
  indices[13] = 2;
  indices[14] = -1;
  // front
  indices[15] = 0;
  indices[16] = 3;
  indices[17] = 4;
  indices[18] = 7;
  indices[19] = -1;
  // back
  indices[20] = 6;
  indices[21] = 5;
  indices[22] = 2;
  indices[23] = 1;
  indices[24] = -1;

  float pts[x.size()][3];
  for(unsigned int i=0; i<x.size(); i++) {
    pts[i][0] = x[i];
    pts[i][1] = y[i];
    pts[i][2] = z[i];
  }

  for(int k=0; k<nz; k++) {
    auto *points = new SoCoordinate3;
    auto *face = new SoIndexedFaceSet;
    points->point.setValues(0, x.size(), pts);
    face->coordIndex.setValues(0, 25, indices);
    soSepRigidBody->addChild(points);
    soSepRigidBody->addChild(face);
    auto *r = new SoRotation;
    soSepRigidBody->addChild(r);
    r->rotation.setValue(SbVec3f(0,1,0),2*M_PI/nz);
  }
  auto *t = new SoTranslation;
  soSepRigidBody->addChild(t);
  t->translation.setValue(SbVec3f(0,-e->getModule()-h/2,0));
  auto *cyl = new SoCylinder;
  soSepRigidBody->addChild(cyl);
  cyl->radius.setValue(r0+w/2);
  cyl->height.setValue(h);
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
    heightEditor->setOpenMBVParameter(e, &OpenMBV::PlanarGear::getHeight, &OpenMBV::PlanarGear::setHeight);
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
