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
      z[nn*j+k] = eta*sin(al)*sin(be)+xi*cos(be);
    }
    s = -w/2;
    for(int k=0; k<nf; k++) {
      double eta = e->getModule()/cos(al)*(double(2*k)/(nf-1)-1);
      double xi = (s-eta*sin(al)*sin(be))/cos(be);
      x[nn*j+nf+k] = -eta*sin(al)*cos(be)+xi*sin(be);
      y[nn*j+nf+k] = signj*eta*cos(al);
      z[nn*j+nf+k] = eta*sin(al)*sin(be)+xi*cos(be);
    }
  }

  int ns = 2*nn;
  int np = nz*ns+2*nz+6;
  float pts[np][3];

  int nii = 4*(nf-1)*5+3*5;
  int ni = nz*nii+25;
  int indices[ni];

  int l=0;
  for(int v=0; v<nz; v++) {
    double x_ = m*M_PI*v ;
    pts[(nz-1)*ns+2*nn+v][0] = x_+m*M_PI/2;
    pts[(nz-1)*ns+2*nn+v][1] = -e->getModule();
    pts[(nz-1)*ns+2*nn+v][2] = w/2;
    pts[(nz-1)*ns+2*nn+nz+v][0] = pts[(nz-1)*ns+2*nn+v][0];
    pts[(nz-1)*ns+2*nn+nz+v][1] = pts[(nz-1)*ns+2*nn+v][1];
    pts[(nz-1)*ns+2*nn+nz+v][2] = -w/2;
    for(int j=0; j<2; j++) {
      int signj=j?-1:1;
      for(int i=nn*j; i<nn*j+nn; i++) {
        pts[v*ns+i][0] = x[i] + x_+signj*dx;
        pts[v*ns+i][1] = y[i];
        pts[v*ns+i][2] = z[i];
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
    indices[l++] = (nz-1)*ns+2*nn+v;
    indices[l++] = (nz-1)*ns+2*nn+nz+v;
    indices[l++] = -1;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = v*ns+4*nf-1;
    indices[l++] = (nz-1)*ns+2*nn+nz+(v==0?nz+1:v-1);
    indices[l++] = (nz-1)*ns+2*nn+(v==0?2*nz:v-1);
    indices[l++] = -1;
  }
  pts[(nz-1)*ns+2*nn+2*nz][0] = -m*M_PI/2;
  pts[(nz-1)*ns+2*nn+2*nz][1] = -e->getModule();
  pts[(nz-1)*ns+2*nn+2*nz][2] = w/2;
  pts[(nz-1)*ns+2*nn+2*nz+1][0] = -m*M_PI/2;
  pts[(nz-1)*ns+2*nn+2*nz+1][1] = -e->getModule();
  pts[(nz-1)*ns+2*nn+2*nz+1][2] = -w/2;
  pts[(nz-1)*ns+2*nn+2*nz+2][0] = pts[(nz-1)*ns+2*nn+2*nz][0];
  pts[(nz-1)*ns+2*nn+2*nz+2][1] = pts[(nz-1)*ns+2*nn+2*nz][1]-h;
  pts[(nz-1)*ns+2*nn+2*nz+2][2] = w/2;
  pts[(nz-1)*ns+2*nn+2*nz+3][0] = pts[(nz-1)*ns+2*nn+2*nz+1][0];
  pts[(nz-1)*ns+2*nn+2*nz+3][1] = pts[(nz-1)*ns+2*nn+2*nz+1][1]-h;
  pts[(nz-1)*ns+2*nn+2*nz+3][2] = -w/2;
  pts[(nz-1)*ns+2*nn+2*nz+4][0] = pts[(nz-1)*ns+2*nn+nz-1][0];
  pts[(nz-1)*ns+2*nn+2*nz+4][1] = pts[(nz-1)*ns+2*nn+nz-1][1]-h;
  pts[(nz-1)*ns+2*nn+2*nz+4][2] = w/2;
  pts[(nz-1)*ns+2*nn+2*nz+5][0] = pts[(nz-1)*ns+2*nn+2*nz-1][0];
  pts[(nz-1)*ns+2*nn+2*nz+5][1] = pts[(nz-1)*ns+2*nn+2*nz-1][1]-h;
  pts[(nz-1)*ns+2*nn+2*nz+5][2] = -w/2;
  indices[l++] = (nz-1)*ns+2*nn+2*nz;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indices[l++] = -1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indices[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indices[l++] = (nz-1)*ns+2*nn+nz-1;
  indices[l++] = -1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indices[l++] = (nz-1)*ns+2*nn+nz-1;
  indices[l++] = -1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indices[l++] = -1;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indices[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indices[l++] = -1;

  auto *points = new SoCoordinate3;
  auto *face = new SoIndexedFaceSet;
  points->point.setValues(0, np, pts);
  face->coordIndex.setValues(0, ni, indices);
  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
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
