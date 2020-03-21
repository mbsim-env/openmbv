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
  double al = e->getPressureAngle();
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();
  double w = e->getWidth();
  double h = e->getHeight()-2*e->getModule();

  double r0 = m*nz/2;
  double dphi = (M_PI/2-b/m)/nz;

  int nf = 2;
  int nn = 2*nf;
  vector<double> x(2*nn), y(2*nn), z(2*nn);
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;
    double s = w/2;
    for(int k=0; k<nf; k++) {
      double eta = e->getModule()/cos(al)*(double(2*k)/(nf-1)-1);
      double xi = (s-eta*sin(al)*sin(be))/cos(be);
      x[nn*j+k] = -eta*sin(al)*cos(be)+xi*sin(be);
      y[nn*j+k] = signj*eta*cos(al);
      z[nn*j+k] = eta*sin(al)*sin(be)+r0+xi*cos(be);
    }
    s = -w/2;
    for(int k=0; k<nf; k++) {
      double eta = e->getModule()/cos(al)*(double(2*k)/(nf-1)-1);
      double xi = (s-eta*sin(al)*sin(be))/cos(be);
      x[nn*j+nf+k] = -eta*sin(al)*cos(be)+xi*sin(be);
      y[nn*j+nf+k] = signj*eta*cos(al);
      z[nn*j+nf+k] = eta*sin(al)*sin(be)+r0+xi*cos(be);
    }
  }

  int ns = 2*nn;
  int np = 2*nz*ns+4*nz;
  float pts[np][3];

  int nii = 4*(nf-1)*5+12*5;
  int ni = nz*nii;
  int indices[ni];

  int l=0;
  for(int v=0; v<nz; v++) {
    double phi = 2*M_PI/nz*v;
    pts[2*((nz-1)*ns+2*nn)+v][0] = sin(phi+M_PI/nz)*(r0+w/2);
    pts[2*((nz-1)*ns+2*nn)+v][1] = -e->getModule();
    pts[2*((nz-1)*ns+2*nn)+v][2] = cos(phi+M_PI/nz)*(r0+w/2);
    pts[2*((nz-1)*ns+2*nn)+nz+v][0] = sin(phi+M_PI/nz)*(r0-w/2);
    pts[2*((nz-1)*ns+2*nn)+nz+v][1] = pts[2*((nz-1)*ns+2*nn)+v][1];
    pts[2*((nz-1)*ns+2*nn)+nz+v][2] = cos(phi+M_PI/nz)*(r0-w/2);
    pts[2*((nz-1)*ns+2*nn)+2*nz+v][0] = pts[2*((nz-1)*ns+2*nn)+v][0];
    pts[2*((nz-1)*ns+2*nn)+2*nz+v][1] = -e->getModule()-h;
    pts[2*((nz-1)*ns+2*nn)+2*nz+v][2] = pts[2*((nz-1)*ns+2*nn)+v][2];
    pts[2*((nz-1)*ns+2*nn)+3*nz+v][0] = pts[2*((nz-1)*ns+2*nn)+nz+v][0];
    pts[2*((nz-1)*ns+2*nn)+3*nz+v][1] = -e->getModule()-h;
    pts[2*((nz-1)*ns+2*nn)+3*nz+v][2] = pts[2*((nz-1)*ns+2*nn)+nz+v][2];
    for(int j=0; j<2; j++) {
      int signj=j?-1:1;
      for(int i=nn*j; i<nn*j+nn; i++) {
        pts[v*ns+i][0] = cos(phi+signj*dphi)*x[i] + sin(phi+signj*dphi)*z[i];
        pts[v*ns+i][1] = y[i];
        pts[v*ns+i][2] = -sin(phi+signj*dphi)*x[i] + cos(phi+signj*dphi)*z[i];
        pts[(nz-1)*ns+2*nn+v*ns+i][0] = pts[v*ns+i][0];
        pts[(nz-1)*ns+2*nn+v*ns+i][1] = pts[v*ns+i][1]-h;
        pts[(nz-1)*ns+2*nn+v*ns+i][2] = pts[v*ns+i][2];
      }
    }

    // left
    for(int k=0; k<nf-1; k++) {
      indices[l++] = v*ns+nf+k;
      indices[l++] = v*ns+nf+k+1;
      indices[l++] = v*ns+k+1;
      indices[l++] = v*ns+k;
      indices[l++] = -1;
    }
    // right
    for(int k=0; k<nf-1; k++) {
      indices[l++] = v*ns+3*nf+k;
      indices[l++] = v*ns+3*nf+k+1;
      indices[l++] = v*ns+2*nf+k+1;
      indices[l++] = v*ns+2*nf+k;
      indices[l++] = -1;
    }
    // top
    indices[l++] = v*ns+3*nf;
    indices[l++] = v*ns+2*nf;
    indices[l++] = v*ns+nf-1;
    indices[l++] = v*ns+2*nf-1;
    indices[l++] = -1;
    // front
    for(int k=0; k<nf-1; k++) {
      indices[l++] = v*ns+k;
      indices[l++] = v*ns+k+1;
      indices[l++] = v*ns+3*nf-(k+2);
      indices[l++] = v*ns+3*nf-(k+1);
      indices[l++] = -1;
    }
    // back
    for(int k=0; k<nf-1; k++) {
      indices[l++] = v*ns+2*nf-(k+1);
      indices[l++] = v*ns+2*nf-(k+2);
      indices[l++] = v*ns+3*nf+k+1;
      indices[l++] = v*ns+3*nf+k;
      indices[l++] = -1;
    }
    //
    indices[l++] = v*ns+nf;
    indices[l++] = v*ns;
    indices[l++] = 2*((nz-1)*ns+2*nn)+v;
    indices[l++] = 2*((nz-1)*ns+2*nn)+nz+v;
    indices[l++] = -1;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = v*ns+4*nf-1;
    indices[l++] = 2*((nz-1)*ns+2*nn)+nz+(v==0?nz-1:v-1);
    indices[l++] = 2*((nz-1)*ns+2*nn)+(v==0?nz-1:v-1);
    indices[l++] = -1;
    //
    indices[l++] = 2*((nz-1)*ns+2*nn)+2*nz+v;
    indices[l++] = 2*((nz-1)*ns+2*nn)+v;
    indices[l++] = v*ns;
    indices[l++] = (nz-1)*ns+2*nn+v*ns;
    indices[l++] = -1;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+3*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns;
    indices[l++] = v*ns;
    indices[l++] = -1;
    indices[l++] = 2*((nz-1)*ns+2*nn)+(v==0?nz-1:v-1);
    indices[l++] = 2*((nz-1)*ns+2*nn)+2*nz+(v==0?nz-1:v-1);
    indices[l++] = (nz-1)*ns+2*nn+v*ns+3*nf-1;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+v*ns+nf;
    indices[l++] = v*ns+nf;
    indices[l++] = 2*((nz-1)*ns+2*nn)+nz+v;
    indices[l++] = 2*((nz-1)*ns+2*nn)+3*nz+v;
    indices[l++] = -1;
    indices[l++] = v*ns+nf;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+nf;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+4*nf-1;
    indices[l++] = v*ns+4*nf-1;
    indices[l++] = -1;
    indices[l++] = v*ns+4*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+4*nf-1;
    indices[l++] = 2*((nz-1)*ns+2*nn)+3*nz+(v==0?nz-1:v-1);
    indices[l++] = 2*((nz-1)*ns+2*nn)+nz+(v==0?nz-1:v-1);
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+v*ns;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+nf;
    indices[l++] = 2*((nz-1)*ns+2*nn)+3*nz+v;
    indices[l++] = 2*((nz-1)*ns+2*nn)+2*nz+v;
    indices[l++] = -1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+4*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+nf;
    indices[l++] = (nz-1)*ns+2*nn+v*ns;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+3*nf-1;
    indices[l++] = -1;
    indices[l++] = 2*((nz-1)*ns+2*nn)+2*nz+(v==0?nz-1:v-1);
    indices[l++] = 2*((nz-1)*ns+2*nn)+3*nz+(v==0?nz-1:v-1);
    indices[l++] = (nz-1)*ns+2*nn+v*ns+4*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+v*ns+3*nf-1;
    indices[l++] = -1;
  }

  auto *points = new SoCoordinate3;
  auto *face = new SoIndexedFaceSet;
  points->point.setValues(0, np, pts);
  face->coordIndex.setValues(0, ni, indices);
  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
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
