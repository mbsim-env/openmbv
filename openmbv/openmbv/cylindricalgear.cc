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
#include "cylindricalgear.h"
#include <Inventor/nodes/SoShapeHints.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/cylindricalgear.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

CylindricalGear::CylindricalGear(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::CylindricalGear>(obj);
  iconFile="cylindricalgear.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int nz = e->getNumberOfTeeth();
  double be = e->getHelixAngle();
  double al = e->getPressureAngle();
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();
  double w = e->getWidth();

  int signe = e->getExternalToothed()?1:-1;
  double r0 = m*nz/2;
  double dphi = (M_PI/2-signe*b/m)/nz;
  double alq = atan(tan(al)/cos(be));

  double phiO[2], phiI[2];
  double d = pow(cos(be),2)+pow(sin(be)*sin(al),2);
  double a = pow(cos(be)*cos(al),2);

  double r = r0*cos(alq);
  double s = w/2;
  b = -2*cos(al)*cos(be)*(s*cos(al)*sin(be)+r0*sin(al));
  double c = s*s*pow(cos(al)*sin(be),2)+(r0*r0-r*r)*d+2*s*r0*sin(al)*sin(be)*cos(al);
  phiO[0] = fabs(b+sqrt(fabs(b*b-4*a*c)))/2/a/r0;

  s = -w/2;
  b = -2*cos(al)*cos(be)*(s*cos(al)*sin(be)+r0*sin(al));
  c = s*s*pow(cos(al)*sin(be),2)+(r0*r0-r*r)*d+2*s*r0*sin(al)*sin(be)*cos(al);
  phiI[0] = fabs(b+sqrt(fabs(b*b-4*a*c)))/2/a/r0;

  r = r0+e->getModule();
  s = w/2;
  b = -2*cos(al)*cos(be)*(s*cos(al)*sin(be)+r0*sin(al));
  c = s*s*pow(cos(al)*sin(be),2)+(r0*r0-r*r)*d+2*s*r0*sin(al)*sin(be)*cos(al);
  phiO[1] = fabs(b+sqrt(fabs(b*b-4*a*c)))/2/a/r0;

  s = -w/2;
  b = -2*cos(al)*cos(be)*(s*cos(al)*sin(be)+r0*sin(al));
  c = s*s*pow(cos(al)*sin(be),2)+(r0*r0-r*r)*d+2*s*r0*sin(al)*sin(be)*cos(al);
  phiI[1] = fabs(b+sqrt(fabs(b*b-4*a*c)))/2/a/r0;

  double etaMaxO[2], etaMaxI[2], etaMinO[2], etaMinI[2];
  etaMinO[0] = -phiO[0];
  etaMinI[0] = -phiI[0];
  etaMaxO[0] = phiO[1];
  etaMaxI[0] = phiI[1];
  etaMinO[1] = -phiI[1];
  etaMinI[1] = -phiO[1];
  etaMaxO[1] = phiI[0];
  etaMaxI[1] = phiO[0];

  int nf = 8;
  int nn = 2*nf;
  double x[2*nn], y[2*nn], z[2*nn];
  for(int j=0; j<2; j++) {
    int signj=j?-1:1;

    double s = w/2;
    for(int k=0; k<nf; k++) {
      double eta = etaMinO[j]+(etaMaxO[j]-etaMinO[j])/(nf-1)*k;
      double xi = signe*(-r0*eta*pow(sin(al),2)*sin(be)+s*cos(be))/(pow(sin(be)*sin(al),2)+pow(cos(be),2));
      double x_ = -r0*eta;
      double l = (x_/cos(be)-signe*xi*tan(be))*sin(al);
      double a = x_-l*sin(al)*cos(be)-signe*xi*sin(be);
      double b = signj*l*cos(al)-r0;
      double c = -l*sin(al)*sin(be)+signe*xi*cos(be);
      x[nn*j+k] = a*cos(eta)-b*sin(eta);
      y[nn*j+k] = a*sin(eta)+b*cos(eta);
      z[nn*j+k] = c;
    }

    s = -w/2;
    for(int k=0; k<nf; k++) {
      double eta = etaMinI[j]+(etaMaxI[j]-etaMinI[j])/(nf-1)*k;
      double xi = signe*(-r0*eta*pow(sin(al),2)*sin(be)+s*cos(be))/(pow(sin(be)*sin(al),2)+pow(cos(be),2));
      double x_ = -r0*eta;
      double l = (x_/cos(be)-signe*xi*tan(be))*sin(al);
      double a = x_-l*sin(al)*cos(be)-signe*xi*sin(be);
      double b = signj*l*cos(al)-r0;
      double c = -l*sin(al)*sin(be)+signe*xi*cos(be);
      x[nn*j+nf+k] = a*cos(eta)-b*sin(eta);
      y[nn*j+nf+k] = a*sin(eta)+b*cos(eta);
      z[nn*j+nf+k] = c;
    }
  }

  SoShapeHints *hints = new SoShapeHints;
  hints->vertexOrdering = e->getExternalToothed()?SoShapeHints::COUNTERCLOCKWISE:SoShapeHints::CLOCKWISE;
  hints->shapeType = SoShapeHints::SOLID;
  soSepRigidBody->addChild(hints);
  auto *points = new SoCoordinate3;
  auto *face = new SoIndexedFaceSet;

  if(e->getExternalToothed()) {
    int ns = 2*nn+4;
    int np = nz*ns+2*nz+2;
    float pts[np][3];

    int nii = 4*(nf-1)*5+7*5+2*6;
    int ni = nz*nii;
    int indices[ni];

    int l=0;
    pts[(nz-1)*ns+2*nn+4+2*nz][0] = 0;
    pts[(nz-1)*ns+2*nn+4+2*nz][1] = 0;
    pts[(nz-1)*ns+2*nn+4+2*nz][2] = w/2;
    pts[(nz-1)*ns+2*nn+5+2*nz][0] = 0;
    pts[(nz-1)*ns+2*nn+5+2*nz][1] = 0;
    pts[(nz-1)*ns+2*nn+5+2*nz][2] = -w/2;
    for(int v=0; v<nz; v++) {
      double phi = 2*M_PI/nz*v;
      pts[(nz-1)*ns+2*nn+4+v][0] = sin(phi-M_PI/nz)*(r0-e->getModule());
      pts[(nz-1)*ns+2*nn+4+v][1] = -cos(phi-M_PI/nz)*(r0-e->getModule());
      pts[(nz-1)*ns+2*nn+4+v][2] = w/2;
      pts[(nz-1)*ns+2*nn+4+nz+v][0] = pts[(nz-1)*ns+2*nn+4+v][0];
      pts[(nz-1)*ns+2*nn+4+nz+v][1] = pts[(nz-1)*ns+2*nn+4+v][1];
      pts[(nz-1)*ns+2*nn+4+nz+v][2] = -w/2;
      for(int j=0; j<2; j++) {
        int signj=j?-1:1;
        for(int i=nn*j; i<nn*j+nn; i++) {
          pts[v*ns+i][0] = cos(phi-signj*dphi)*x[i] - sin(phi-signj*dphi)*y[i];
          pts[v*ns+i][1] = sin(phi-signj*dphi)*x[i] + cos(phi-signj*dphi)*y[i];
          pts[v*ns+i][2] = z[i];
        }
        pts[v*ns+2*nn+2*j][0] = pts[v*ns+(3*nf-1)*j][0]*(r0-e->getModule())/(r0*cos(alq));
        pts[v*ns+2*nn+2*j][1] = pts[v*ns+(3*nf-1)*j][1]*(r0-e->getModule())/(r0*cos(alq));
        pts[v*ns+2*nn+2*j][2] = w/2;
        pts[v*ns+2*nn+2*j+1][0] = pts[v*ns+2*nn+2*j][0];
        pts[v*ns+2*nn+2*j+1][1] = pts[v*ns+2*nn+2*j][1];
        pts[v*ns+2*nn+2*j+1][2] = -w/2;
      }

      // left
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = v*ns+nf;
      indices[l++] = v*ns+0;
      indices[l++] = v*ns+2*nn;
      indices[l++] = -1;
      for(int k=0; k<nf-1; k++) {
        indices[l++] = v*ns+nf+k;
        indices[l++] = v*ns+nf+k+1;
        indices[l++] = v*ns+k+1;
        indices[l++] = v*ns+k;
        indices[l++] = -1;
      }
      // right
      indices[l++] = v*ns+4*nf-1;
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = v*ns+3*nf-1;
      indices[l++] = -1;
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
      indices[l++] = v*ns+2*nn;
      indices[l++] = v*ns;
      indices[l++] = v*ns+3*nf-1;
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = -1;
      indices[l++] = (nz-1)*ns+2*nn+4+2*nz;
      indices[l++] = (nz-1)*ns+2*nn+4+v;
      indices[l++] = v*ns+2*nn;
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = (nz-1)*ns+2*nn+4+(v==(nz-1)?0:v+1);
      indices[l++] = -1;
      // back
      for(int k=0; k<nf-1; k++) {
        indices[l++] = v*ns+2*nf-(k+1);
        indices[l++] = v*ns+2*nf-(k+2);
        indices[l++] = v*ns+3*nf+k+1;
        indices[l++] = v*ns+3*nf+k;
        indices[l++] = -1;
      }
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = v*ns+4*nf-1;
      indices[l++] = v*ns+nf;
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = -1;
      indices[l++] = (nz-1)*ns+2*nn+5+2*nz;
      indices[l++] = (nz-1)*ns+2*nn+4+nz+(v==(nz-1)?0:v+1);
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = (nz-1)*ns+2*nn+4+nz+v;
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
    }
    points->point.setValues(0, np, pts);
    face->coordIndex.setValues(0, ni, indices);
  }
  else {
    int ns = 2*nn+8;
    int np = nz*ns+4*nz;
    float pts[np][3];

    int nii = 2*(nf-1)*5+14*5+4*nf*4;
    int ni = nz*nii;
    int indices[ni];

    int l=0;
    for(int v=0; v<nz; v++) {
      double phi = 2*M_PI/nz*v;
      pts[(nz-1)*ns+2*nn+8+v][0] = sin(phi-M_PI/nz)*r0*cos(alq);
      pts[(nz-1)*ns+2*nn+8+v][1] = -cos(phi-M_PI/nz)*r0*cos(alq);
      pts[(nz-1)*ns+2*nn+8+v][2] = w/2;
      pts[(nz-1)*ns+2*nn+8+nz+v][0] = pts[(nz-1)*ns+2*nn+8+v][0];
      pts[(nz-1)*ns+2*nn+8+nz+v][1] = pts[(nz-1)*ns+2*nn+8+v][1];
      pts[(nz-1)*ns+2*nn+8+nz+v][2] = -w/2;
      //
      pts[(nz-1)*ns+2*nn+8+2*nz+v][0] = sin(phi-M_PI/nz)*(r0+e->getModule()*2);
      pts[(nz-1)*ns+2*nn+8+2*nz+v][1] = -cos(phi-M_PI/nz)*(r0+e->getModule()*2);
      pts[(nz-1)*ns+2*nn+8+2*nz+v][2] = w/2;
      pts[(nz-1)*ns+2*nn+8+3*nz+v][0] = pts[(nz-1)*ns+2*nn+8+2*nz+v][0];
      pts[(nz-1)*ns+2*nn+8+3*nz+v][1] = pts[(nz-1)*ns+2*nn+8+2*nz+v][1];
      pts[(nz-1)*ns+2*nn+8+3*nz+v][2] = -w/2;
      for(int j=0; j<2; j++) {
        int signj=j?-1:1;
        for(int i=nn*j; i<nn*j+nn; i++) {
          pts[v*ns+i][0] = cos(phi-signj*dphi)*x[i] - sin(phi-signj*dphi)*y[i];
          pts[v*ns+i][1] = sin(phi-signj*dphi)*x[i] + cos(phi-signj*dphi)*y[i];
          pts[v*ns+i][2] = z[i];
        }
        pts[v*ns+2*nn+4*j][0] = pts[v*ns+(3*nf-1)*j][0]*(r0+e->getModule()*2)/(r0*cos(alq));
        pts[v*ns+2*nn+4*j][1] = pts[v*ns+(3*nf-1)*j][1]*(r0+e->getModule()*2)/(r0*cos(alq));
        pts[v*ns+2*nn+4*j][2] = w/2;
        pts[v*ns+2*nn+4*j+1][0] = pts[v*ns+2*nn+4*j][0];
        pts[v*ns+2*nn+4*j+1][1] = pts[v*ns+2*nn+4*j][1];
        pts[v*ns+2*nn+4*j+1][2] = -w/2;
        pts[v*ns+2*nn+4*j+2][0] = pts[v*ns+nf-1+(nf+1)*j][0]*(r0+e->getModule()*2)/(r0+e->getModule());
        pts[v*ns+2*nn+4*j+2][1] = pts[v*ns+nf-1+(nf+1)*j][1]*(r0+e->getModule()*2)/(r0+e->getModule());
        pts[v*ns+2*nn+4*j+2][2] = w/2;
        pts[v*ns+2*nn+4*j+3][0] = pts[v*ns+2*nn+4*j+2][0];
        pts[v*ns+2*nn+4*j+3][1] = pts[v*ns+2*nn+4*j+2][1];
        pts[v*ns+2*nn+4*j+3][2] = -w/2;
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
      indices[l++] = (nz-1)*ns+2*nn+8+nz+v;
      indices[l++] = v*ns+nf;
      indices[l++] = v*ns;
      indices[l++] = (nz-1)*ns+2*nn+8+v;
      indices[l++] = -1;
      //
      indices[l++] = (nz-1)*ns+2*nn+8+(v==(nz-1)?0:v+1);
      indices[l++] = v*ns+3*nf-1;
      indices[l++] = v*ns+3*nf-1+nf;
      indices[l++] = (nz-1)*ns+2*nn+8+nz+(v==(nz-1)?0:v+1);
      indices[l++] = -1;
      // front
      indices[l++] = (nz-1)*ns+2*nn+8+2*nz+v;
      indices[l++] = (nz-1)*ns+2*nn+8+v;
      indices[l++] = v*ns;
      indices[l++] = v*ns+2*nn;
      indices[l++] = -1;
      //
      for(int i=0; i<nf-1; i++) {
        indices[l++] = v*ns+i+1;
        indices[l++] = v*ns+2*nn;
        indices[l++] = v*ns+i;
        indices[l++] = -1;
      }
      indices[l++] = v*ns+nf-1;
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = v*ns+2*nn;
      indices[l++] = -1;
      //
      for(int i=0; i<nf-1; i++) {
        indices[l++] = v*ns+2*nf+i+1;
        indices[l++] = v*ns+2*nn+4;
        indices[l++] = v*ns+2*nf+i;
        indices[l++] = -1;
      }
      indices[l++] = v*ns+2*nn+4;
      indices[l++] = v*ns+2*nn+6;
      indices[l++] = v*ns+2*nf;
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = v*ns+nf-1;
      indices[l++] = v*ns+2*nf;
      indices[l++] = v*ns+2*nn+6;
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+4;
      indices[l++] = v*ns+3*nf-1;
      indices[l++] = (nz-1)*ns+2*nn+8+(v==(nz-1)?0:v+1);
      indices[l++] = (nz-1)*ns+2*nn+8+2*nz+(v==(nz-1)?0:v+1);
      indices[l++] = -1;
      // outside
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = v*ns+2*nn+6;
      indices[l++] = v*ns+2*nn+7;
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+4;
      indices[l++] = v*ns+2*nn+5;
      indices[l++] = v*ns+2*nn+7;
      indices[l++] = v*ns+2*nn+6;
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+5;
      indices[l++] = v*ns+2*nn+4;
      indices[l++] = (nz-1)*ns+2*nn+8+2*nz+(v==(nz-1)?0:v+1);
      indices[l++] = (nz-1)*ns+2*nn+8+3*nz+(v==(nz-1)?0:v+1);
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+2;
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = v*ns+2*nn;
      indices[l++] = -1;
      //
      indices[l++] = (nz-1)*ns+2*nn+8+3*nz+v;
      indices[l++] = (nz-1)*ns+2*nn+8+2*nz+v;
      indices[l++] = v*ns+2*nn;
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = -1;
      // back
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = v*ns+nf;
      indices[l++] = (nz-1)*ns+2*nn+8+nz+v;
      indices[l++] = (nz-1)*ns+2*nn+8+3*nz+v;
      indices[l++] = -1;
      //
      for(int i=0; i<nf-1; i++) {
        indices[l++] = v*ns+nf+i;
        indices[l++] = v*ns+2*nn+1;
        indices[l++] = v*ns+nf+i+1;
        indices[l++] = -1;
      }
      indices[l++] = v*ns+2*nn+1;
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = v*ns+2*nf-1;
      indices[l++] = -1;
      //
      for(int i=0; i<nf-1; i++) {
        indices[l++] = v*ns+3*nf+i;
        indices[l++] = v*ns+2*nn+5;
        indices[l++] = v*ns+3*nf+i+1;
        indices[l++] = -1;
      }
      indices[l++] = v*ns+3*nf;
      indices[l++] = v*ns+2*nn+7;
      indices[l++] = v*ns+2*nn+5;
      indices[l++] = -1;
      //
      indices[l++] = v*ns+2*nn+7;
      indices[l++] = v*ns+3*nf;
      indices[l++] = v*ns+2*nf-1;
      indices[l++] = v*ns+2*nn+3;
      indices[l++] = -1;
      //
      indices[l++] = (nz-1)*ns+2*nn+8+3*nz+(v==(nz-1)?0:v+1);
      indices[l++] = (nz-1)*ns+2*nn+8+nz+(v==(nz-1)?0:v+1);
      indices[l++] = v*ns+4*nf-1;
      indices[l++] = v*ns+2*nn+5;
      indices[l++] = -1;
    }
    points->point.setValues(0, np, pts);
    face->coordIndex.setValues(0, ni, indices);
  }

  soSepRigidBody->addChild(points);
  soSepRigidBody->addChild(face);
}
 
void CylindricalGear::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(5, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getNumberOfTeeth, &OpenMBV::CylindricalGear::setNumberOfTeeth);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getWidth, &OpenMBV::CylindricalGear::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getHelixAngle, &OpenMBV::CylindricalGear::setHelixAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getModule, &OpenMBV::CylindricalGear::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getPressureAngle, &OpenMBV::CylindricalGear::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getBacklash, &OpenMBV::CylindricalGear::setBacklash);
    BoolEditor *solidEditor=new BoolEditor(properties, QIcon(), "External thoothed", "CylindricalGear::externalToothed");
    solidEditor->setOpenMBVParameter(e, &OpenMBV::CylindricalGear::getExternalToothed, &OpenMBV::CylindricalGear::setExternalToothed);
  }
}

}
