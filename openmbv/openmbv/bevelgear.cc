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
#include <Inventor/nodes/SoShapeHints.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/bevelgear.h"
#include <QMenu>
#include <cfloat>
#include <boost/math/tools/roots.hpp>

using namespace std;
using namespace boost::math::tools;

namespace OpenMBVGUI {

struct Residuum {
  Residuum(double al_, double be_, double ga_, double r1_, double s_, double h_, int signi_) : al(al_), be(be_), ga(ga_), r1(r1_), s(s_), h(h_), signi(signi_) { }
  double operator()(const double &eta) {
    double phi = -sin(ga)*eta;
    double xi = (s*cos(phi-be)+r1*sin(phi)*pow(sin(al),2)*sin(be))/(-sin(phi-be)*pow(sin(al),2)*sin(be)+cos(phi-be)*cos(be));
    double l = (sin(phi)/cos(phi-be)*r1+xi*tan(phi-be))*sin(al);
    double x = -l*sin(al)*cos(phi-be)+xi*sin(phi-be)+r1*sin(phi);
    double b = signi*l*cos(al);
    double c = l*sin(al)*sin(phi-be)+xi*cos(phi-be)+r1*cos(phi);
    double y = b*cos(ga)-c*sin(ga);
    double z = b*sin(ga)+c*cos(ga);
    double hm = z*tan(ga)+h/cos(ga);
    return x*x+y*y-hm*hm;
  }
  double al, be, ga, r1, s, h;
  int signi;
};

BevelGear::BevelGear(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::BevelGear>(obj);
  iconFile="bevelgear.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int nz = e->getNumberOfTeeth();
  double be = e->getHelixAngle();
  double ga = e->getPitchAngle();
  double al = e->getPressureAngle();
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();
  double w = e->getWidth();

  double r0 = m*nz/2;
  double r1 = r0/sin(ga);
  double d = sqrt(r1*r1-r0*r0);
  double dphi = (M_PI/2-b/m)/nz;

  double etaMaxO[2], etaMaxI[2], etaMinO[2], etaMinI[2];
  const boost::uintmax_t maxit = 20;
  boost::uintmax_t it = maxit;
  int digits = std::numeric_limits<double>::digits;
  int get_digits = digits - 3;
  double h[2];
  h[0] = -e->getModule()/2;
  h[1] = e->getModule();
  eps_tolerance<double> tol(get_digits);
  for(int i=0; i<2; i++) {
    int signi=i?-1:1;
    bool rising = i?false:true;
    std::pair<double, double> r = bracket_and_solve_root(Residuum(al,be,ga,r1,w/2,h[i],signi),-0.5,2.,rising,tol,it);
    etaMinO[i] = r.first;
    r = bracket_and_solve_root(Residuum(al,be,ga,r1,-w/2,h[i],signi),-0.5,2.,rising,tol,it);
    etaMinI[i] = r.first;
    r = bracket_and_solve_root(Residuum(al,be,ga,r1,w/2,h[not i],signi),0.5,2.,rising,tol,it);
    etaMaxO[i] = r.first;
    r = bracket_and_solve_root(Residuum(al,be,ga,r1,-w/2,h[not i],signi),0.5,2.,rising,tol,it);
    etaMaxI[i] = r.first;
  }

  int nf = 8;
  int nn = 2*nf;
  double x[2*nn], y[2*nn], z[2*nn];
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;

    double s = w/2;
    for(int k=0; k<nf; k++) {
      double eta = etaMinO[j]+(etaMaxO[j]-etaMinO[j])/(nf-1)*k;
      double phi = -sin(ga)*eta;
      double xi = (s*cos(phi-be)+r1*sin(phi)*pow(sin(al),2)*sin(be))/(-sin(phi-be)*pow(sin(al),2)*sin(be)+cos(phi-be)*cos(be));
      double l = (sin(phi)/cos(phi-be)*r1+xi*tan(phi-be))*sin(al);
      double a = -l*sin(al)*cos(phi-be)+xi*sin(phi-be)+r1*sin(phi);
      double b = signj*l*cos(al);
      double c = l*sin(al)*sin(phi-be)+xi*cos(phi-be)+r1*cos(phi);
      x[nn*j+k] = a*cos(eta)-b*sin(eta)*cos(ga)+c*sin(eta)*sin(ga);
      y[nn*j+k] = a*sin(eta)+b*cos(eta)*cos(ga)-c*cos(eta)*sin(ga);
      z[nn*j+k] = b*sin(ga)+c*cos(ga)-d;
    }

    s = -w/2;
    for(int k=0; k<nf; k++) {
      double eta = etaMinI[j]+(etaMaxI[j]-etaMinI[j])/(nf-1)*k;
      double phi = -sin(ga)*eta;
      double xi = (s*cos(phi-be)+r1*sin(phi)*pow(sin(al),2)*sin(be))/(-sin(phi-be)*pow(sin(al),2)*sin(be)+cos(phi-be)*cos(be));
      double l = (sin(phi)/cos(phi-be)*r1+xi*tan(phi-be))*sin(al);
      double a = -l*sin(al)*cos(phi-be)+xi*sin(phi-be)+r1*sin(phi);
      double b = signj*l*cos(al);
      double c = l*sin(al)*sin(phi-be)+xi*cos(phi-be)+r1*cos(phi);
      x[nn*j+nf+k] = a*cos(eta)-b*sin(eta)*cos(ga)+c*sin(eta)*sin(ga);
      y[nn*j+nf+k] = a*sin(eta)+b*cos(eta)*cos(ga)-c*cos(eta)*sin(ga);
      z[nn*j+nf+k] = b*sin(ga)+c*cos(ga)-d;
    }
  }

  auto *points = new SoCoordinate3;
  auto *face = new SoIndexedFaceSet;

  int ns = 2*nn+4;
  int np = nz*ns+2*nz+2;
  float pts[np][3];

  int nii = 4*(nf-1)*5+7*5+6*4;
  int ni = nz*nii;
  int indices[ni];

  int l=0;
  pts[(nz-1)*ns+2*nn+4+2*nz][0] = 0;
  pts[(nz-1)*ns+2*nn+4+2*nz][1] = 0;
  pts[(nz-1)*ns+2*nn+4+2*nz][2] = e->getModule()*sin(ga)-d+(r1+w/2)*cos(ga);
  pts[(nz-1)*ns+2*nn+5+2*nz][0] = 0;
  pts[(nz-1)*ns+2*nn+5+2*nz][1] = 0;
  pts[(nz-1)*ns+2*nn+5+2*nz][2] = e->getModule()*sin(ga)-d+(r1-w/2)*cos(ga);
  for(int v=0; v<nz; v++) {
    double phi = 2*M_PI/nz*v;
    pts[(nz-1)*ns+2*nn+4+v][0] = -sin(phi-M_PI/nz)*(e->getModule()*cos(ga)-(r1+w/2)*sin(ga));
    pts[(nz-1)*ns+2*nn+4+v][1] = cos(phi-M_PI/nz)*(e->getModule()*cos(ga)-(r1+w/2)*sin(ga));
    pts[(nz-1)*ns+2*nn+4+v][2] = e->getModule()*sin(ga)-d+(r1+w/2)*cos(ga);
    pts[(nz-1)*ns+2*nn+4+nz+v][0] = -sin(phi-M_PI/nz)*(e->getModule()*cos(ga)-(r1-w/2)*sin(ga));
    pts[(nz-1)*ns+2*nn+4+nz+v][1] = cos(phi-M_PI/nz)*(e->getModule()*cos(ga)-(r1-w/2)*sin(ga));
    pts[(nz-1)*ns+2*nn+4+nz+v][2] = e->getModule()*sin(ga)-d+(r1-w/2)*cos(ga);
    for(int j=0; j<2; j++) {
      int signj=j?-1:1;
      for(int i=nn*j; i<nn*j+nn; i++) {
        pts[v*ns+i][0] = cos(phi-signj*dphi)*x[i] - sin(phi-signj*dphi)*y[i];
        pts[v*ns+i][1] = sin(phi-signj*dphi)*x[i] + cos(phi-signj*dphi)*y[i];
        pts[v*ns+i][2] = z[i];
      }
      double R1 = (pts[v*ns+(3*nf-1)*j][2]+d)*tan(ga)-e->getModule()/2/cos(ga);
      double R2 = (e->getModule()*sin(ga)+(r1+w/2)*cos(ga))*tan(ga)-e->getModule()/cos(ga);
      pts[v*ns+2*nn+2*j][0] = pts[v*ns+(3*nf-1)*j][0]*R2/R1;
      pts[v*ns+2*nn+2*j][1] = pts[v*ns+(3*nf-1)*j][1]*R2/R1;
      pts[v*ns+2*nn+2*j][2] = e->getModule()*sin(ga)-d+(r1+w/2)*cos(ga);
      R1 = (pts[v*ns+nf+(3*nf-1)*j][2]+d)*tan(ga)-e->getModule()/2/cos(ga);
      R2 = (e->getModule()*sin(ga)+(r1-w/2)*cos(ga))*tan(ga)-e->getModule()/cos(ga);
      pts[v*ns+2*nn+2*j+1][0] = pts[v*ns+nf+(3*nf-1)*j][0]*R2/R1;
      pts[v*ns+2*nn+2*j+1][1] = pts[v*ns+nf+(3*nf-1)*j][1]*R2/R1;
      pts[v*ns+2*nn+2*j+1][2] = e->getModule()*sin(ga)-d+(r1-w/2)*cos(ga);
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
    //
    indices[l++] = (nz-1)*ns+2*nn+4+nz+v;
    indices[l++] = v*ns+2*nn+1;
    indices[l++] = v*ns+2*nn;
    indices[l++] = (nz-1)*ns+2*nn+4+v;
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+4+(v==(nz-1)?0:v+1);
    indices[l++] = v*ns+2*nn+2;
    indices[l++] = v*ns+2*nn+3;
    indices[l++] = (nz-1)*ns+2*nn+4+nz+(v==(nz-1)?0:v+1);
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
    indices[l++] = (nz-1)*ns+2*nn+4+2*nz;
    indices[l++] = (nz-1)*ns+2*nn+4+v;
    indices[l++] = v*ns+2*nn;
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+4+2*nz;
    indices[l++] = v*ns+2*nn;
    indices[l++] = v*ns+2*nn+2;
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+4+2*nz;
    indices[l++] = v*ns+2*nn+2;
    indices[l++] = (nz-1)*ns+2*nn+4+(v==(nz-1)?0:v+1);
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+5+2*nz;
    indices[l++] = v*ns+2*nn+1;
    indices[l++] = (nz-1)*ns+2*nn+4+nz+v;
    indices[l++] = -1;
    //
    indices[l++] = v*ns+2*nn+3;
    indices[l++] = v*ns+2*nn+1;
    indices[l++] = (nz-1)*ns+2*nn+5+2*nz;
    indices[l++] = -1;
    //
    indices[l++] = (nz-1)*ns+2*nn+4+nz+(v==(nz-1)?0:v+1);
    indices[l++] = v*ns+2*nn+3;
    indices[l++] = (nz-1)*ns+2*nn+5+2*nz;
    indices[l++] = -1;

    indices[l++] = v*ns+2*nn+2;
    indices[l++] = v*ns+2*nn;
    indices[l++] = v*ns;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = -1;

    indices[l++] = v*ns+2*nn+1;
    indices[l++] = v*ns+2*nn+3;
    indices[l++] = v*ns+3*nf-1+nf;
    indices[l++] = v*ns+nf;
    indices[l++] = -1;

    indices[l++] = v*ns+nf;
    indices[l++] = v*ns;
    indices[l++] = v*ns+2*nn;
    indices[l++] = v*ns+2*nn+1;
    indices[l++] = -1;

    indices[l++] = v*ns+3*nf-1+nf;
    indices[l++] = v*ns+2*nn+3;
    indices[l++] = v*ns+2*nn+2;
    indices[l++] = v*ns+3*nf-1;
    indices[l++] = -1;
  }
  points->point.setValues(0, np, pts);
  face->coordIndex.setValues(0, ni, indices);

  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
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
