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
#include "rack.h"
#include <Inventor/nodes/SoCube.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/rack.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Rack::Rack(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::Rack>(obj);
  iconFile="rack.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int nz = e->getNumberOfTeeth();
  double be = e->getHelixAngle();
  double al = e->getPressureAngle();
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();
  double w = e->getWidth();
  double h = e->getHeight()-2*e->getModule();

  double dx = (m*M_PI/2-b)/2;

  vector<double> x(8), y(8), z(8);
  float pts[x.size()][3];
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;
    double eta = -e->getModule()/cos(al);
    double s = w/2;
    double xi = (s-eta*sin(al)*sin(be))/cos(be);
    x[4*j+0] = -eta*sin(al)*cos(be)+xi*sin(be);
    y[4*j+0] = signj*eta*cos(al);
    z[4*j+0] = eta*sin(al)*sin(be)+xi*cos(be);
    s = -w/2;
    xi = (s-eta*sin(al)*sin(be))/cos(be);
    x[4*j+1] = -eta*sin(al)*cos(be)+xi*sin(be);
    y[4*j+1] = signj*eta*cos(al);
    z[4*j+1] = eta*sin(al)*sin(be)+xi*cos(be);
    eta = e->getModule()/cos(al);
    s = -w/2;
    xi = (s-eta*sin(al)*sin(be))/cos(be);
    x[4*j+2] = -eta*sin(al)*cos(be)+xi*sin(be);
    y[4*j+2] = signj*eta*cos(al);
    z[4*j+2] = eta*sin(al)*sin(be)+xi*cos(be);
    s = w/2;
    xi = (s-eta*sin(al)*sin(be))/cos(be);
    x[4*j+3] = -eta*sin(al)*cos(be)+xi*sin(be);
    y[4*j+3] = signj*eta*cos(al);
    z[4*j+3] = eta*sin(al)*sin(be)+xi*cos(be);
    for(int i=4*j; i<4*j+4; i++) {
      pts[i][0] = x[i] + signj*dx;
      pts[i][1] = y[i];
      pts[i][2] = z[i];
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

  for(int k=0; k<nz; k++) {
    auto *points = new SoCoordinate3;
    auto *face = new SoIndexedFaceSet;
    points->point.setValues(0, x.size(), pts);
    face->coordIndex.setValues(0, 25, indices);
    soSepRigidBody->addChild(points);
    soSepRigidBody->addChild(face);
    auto *t = new SoTranslation;
    soSepRigidBody->addChild(t);
    t->translation.setValue(SbVec3f(m*M_PI,0,0));
  }
  auto *t = new SoTranslation;
  soSepRigidBody->addChild(t);
  t->translation.setValue(SbVec3f(-(2*nz+2)*m*M_PI/4,-e->getModule()-h/2,0));
  auto *cube = new SoCube;
  soSepRigidBody->addChild(cube);
  cube->width.setValue(nz*m*M_PI);
  cube->height.setValue(h);
  cube->depth.setValue(w);
}
 
void Rack::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(1, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getNumberOfTeeth, &OpenMBV::Rack::setNumberOfTeeth);
    FloatEditor *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getHeight, &OpenMBV::Rack::setHeight);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getWidth, &OpenMBV::Rack::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getHelixAngle, &OpenMBV::Rack::setHelixAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getModule, &OpenMBV::Rack::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getPressureAngle, &OpenMBV::Rack::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getBacklash, &OpenMBV::Rack::setBacklash);
  }
}

}
