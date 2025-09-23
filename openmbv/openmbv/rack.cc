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
  //iconFile="rack.svg";
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
  int indf[ni];

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
      indf[l++] = v*ns+nf+k;
      indf[l++] = v*ns+nf+k+1;
      indf[l++] = v*ns+k+1;
      indf[l++] = v*ns+k;
      indf[l++] = -1;
    }
    // right
    for(int k=0; k<nf-1; k++) {
      indf[l++] = v*ns+3*nf+k;
      indf[l++] = v*ns+3*nf+k+1;
      indf[l++] = v*ns+2*nf+k+1;
      indf[l++] = v*ns+2*nf+k;
      indf[l++] = -1;
    }
    // top
    indf[l++] = v*ns+3*nf;
    indf[l++] = v*ns+2*nf;
    indf[l++] = v*ns+nf-1;
    indf[l++] = v*ns+2*nf-1;
    indf[l++] = -1;
    // front
    for(int k=0; k<nf-1; k++) {
      indf[l++] = v*ns+k;
      indf[l++] = v*ns+k+1;
      indf[l++] = v*ns+3*nf-(k+2);
      indf[l++] = v*ns+3*nf-(k+1);
      indf[l++] = -1;
    }
    // back
    for(int k=0; k<nf-1; k++) {
      indf[l++] = v*ns+2*nf-(k+1);
      indf[l++] = v*ns+2*nf-(k+2);
      indf[l++] = v*ns+3*nf+k+1;
      indf[l++] = v*ns+3*nf+k;
      indf[l++] = -1;
    }
    //
    indf[l++] = v*ns+nf;
    indf[l++] = v*ns;
    indf[l++] = (nz-1)*ns+2*nn+v;
    indf[l++] = (nz-1)*ns+2*nn+nz+v;
    indf[l++] = -1;
    indf[l++] = v*ns+3*nf-1;
    indf[l++] = v*ns+4*nf-1;
    indf[l++] = (nz-1)*ns+2*nn+nz+(v==0?nz+1:v-1);
    indf[l++] = (nz-1)*ns+2*nn+(v==0?2*nz:v-1);
    indf[l++] = -1;
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
  indf[l++] = (nz-1)*ns+2*nn+2*nz;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indf[l++] = -1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indf[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indf[l++] = (nz-1)*ns+2*nn+nz-1;
  indf[l++] = -1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indf[l++] = (nz-1)*ns+2*nn+nz-1;
  indf[l++] = -1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indf[l++] = -1;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indf[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indf[l++] = -1;

  l = 0;
  const int nl = 26*nz+22;
  int indl[nl];
  for(int v=0; v<nz; v++) {
    indl[l++] = (nz-1)*ns+2*nn+v;
    indl[l++] = v*ns;
    indl[l++] = v*ns+nf-1;
    indl[l++] = v*ns+nn;
    indl[l++] = v*ns+nn+nf-1;
    indl[l++] = (nz-1)*ns+2*nn+(v==0?2*nz:v-1);
    indl[l++] = -1;
    indl[l++] = (nz-1)*ns+2*nn+nz+v;
    indl[l++] = v*ns+nf;
    indl[l++] = v*ns+2*nf-1;
    indl[l++] = v*ns+nn+nf;
    indl[l++] = v*ns+nn+2*nf-1;
    indl[l++] = (nz-1)*ns+2*nn+nz+(v==0?nz+1:v-1);
    indl[l++] = -1;
    indl[l++] = v*ns;
    indl[l++] = v*ns+nf;
    indl[l++] = -1;
    indl[l++] = v*ns+nf-1;
    indl[l++] = v*ns+2*nf-1;
    indl[l++] = -1;
    indl[l++] = v*ns+nn;
    indl[l++] = v*ns+nn+nf;
    indl[l++] = -1;
    indl[l++] = v*ns+nn+nf-1;
    indl[l++] = v*ns+nn+2*nf-1;
    indl[l++] = -1;
  }
  indl[l++] = (nz-1)*ns+2*nn+2*nz;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indl[l++] = (nz-1)*ns+2*nn+nz-1;
  indl[l++] = -1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indl[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indl[l++] = -1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+1;
  indl[l++] = -1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+2;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+3;
  indl[l++] = -1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+4;
  indl[l++] = (nz-1)*ns+2*nn+2*nz+5;
  indl[l++] = -1;
  indl[l++] = (nz-1)*ns+2*nn+nz-1;
  indl[l++] = (nz-1)*ns+2*nn+2*nz-1;
  indl[l++] = -1;

  auto *points = new SoCoordinate3;
  points->point.setValues(0, np, pts);

  auto *line = new SoIndexedLineSet;
  line->coordIndex.setValues(0, nl, indl);

  auto *face = new SoIndexedFaceSet;
  face->coordIndex.setValues(0, ni, indf);

  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
  soSepRigidBody->addChild(soOutLineSwitch);
  soOutLineSep->addChild(line);
}
 
void Rack::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    auto *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(1, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getNumberOfTeeth, &OpenMBV::Rack::setNumberOfTeeth);
    auto *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getHeight, &OpenMBV::Rack::setHeight);
    auto *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getWidth, &OpenMBV::Rack::setWidth);
    auto *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getHelixAngle, &OpenMBV::Rack::setHelixAngle);
    auto *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getModule, &OpenMBV::Rack::setModule);
    auto *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getPressureAngle, &OpenMBV::Rack::setPressureAngle);
    auto *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::Rack::getBacklash, &OpenMBV::Rack::setBacklash);
  }
}

}
