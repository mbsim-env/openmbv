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
#include "gearwheel.h"
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/gearwheel.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

GearWheel::GearWheel(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::GearWheel>(obj);
  iconFile="gearwheel.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int z = e->getNumberOfTeeth();
  double width = e->getWidth();
  double be = e->getHelixAngle();
  double ga = e->getPitchAngle();
  double al0 = e->getPressureAngle(); 
  double m = e->getModule();
  double b = e->getBacklash();

  double d0 = m*z;
  double p0 = M_PI*d0/z;
  double c = 0.167*m;
  double s0 = p0/2;
  double df = d0 - 2*(m+c);
  double da = d0 + 2*m;
  double db = d0*cos(al0);
  double phi0 = tan(al0)- al0;
  double ala = acos(db/da);
  double phia = tan(ala) - ala;
  double sb = db*(s0/d0+phi0)-b;
  double deb = 2*sb/db;
  double ga0 = 2*p0/d0;
  double rb = db/2;
  double ra = da/2;
  double rf = df/2;
  int numb = 5;
  double dphi = (ga0-deb)/(2*numb);
  double phi = -ga0/2;
  vector<double> x(6*numb-1), y(6*numb-1);
  for (int i=0; i<numb; i++) {
    x[i] = rf*sin(phi);
    y[i] = rf*cos(phi);
    phi += dphi;
  }
  dphi = (phia+ala)/numb;
  phi = 0;
  for (int i=numb; i<2*numb; i++) {
    x[i] = rb*(cos(-deb/2)*(sin(phi)-cos(phi)*phi) + sin(-deb/2)*(cos(phi)+sin(phi)*phi));
    y[i] = rb*(-sin(-deb/2)*(sin(phi)-cos(phi)*phi) + cos(-deb/2)*(cos(phi)+sin(phi)*phi));
    phi += dphi;
  }
  dphi = (deb/2 - phia)/numb;
  phi = phia - deb/2;
  for (int i=2*numb; i<3*numb; i++) {
    x[i] = ra*sin(phi);
    y[i] = ra*cos(phi);
    phi += dphi;
  }
  for (int i=6*numb-2, j=1; i>=3*numb; i--,j++) {
    x[i] = -x[j];
    y[i] = y[j];
  }
  vector<double> X(z*x.size());
  vector<double> Y(z*x.size());
  for(int i=0; i<z; i++) {
    int k = i*x.size();
    for(unsigned int j=0; j<x.size(); j++) {
      X[k+j] = cos(i*ga0)*x[j] + sin(i*ga0)*y[j];
      Y[k+j] = -sin(i*ga0)*x[j] + cos(i*ga0)*y[j];
    }
  }

  auto *t = new SoTranslation;
  soSepRigidBody->addChild(t);
  t->translation.setValue(0,0,width/2);

  auto *r = new SoRotation;
  soSepRigidBody->addChild(r);
  r->rotation.setValue(SbVec3f(1,0,0),-M_PI/2);

  auto *extrusion = new SoVRMLExtrusion;
  soSepRigidBody->addChild(extrusion);

  // cross section
  extrusion->crossSection.setNum(X.size()+1);
  SbVec2f *cs = extrusion->crossSection.startEditing();
  for(size_t i=0;i<X.size();i++) cs[i] = SbVec2f(X[i], Y[i]); // clockwise in local coordinate system
  cs[X.size()] =  SbVec2f(X[0], Y[0]); // closed cross section
  extrusion->crossSection.finishEditing();
  extrusion->crossSection.setDefault(FALSE);

  // set spine
  int numw = 5;
  double dw = width/numw;
  extrusion->spine.setNum(numw+1);
  SbVec3f *sp = extrusion->spine.startEditing();
  for(int i=0; i<=numw; i++)
    sp[i] = SbVec3f(0,i*dw,0);
  extrusion->spine.finishEditing();
  extrusion->spine.setDefault(FALSE);

  // set helix angle
  dphi = asin(width*tan(be)/rb)/numw;
  extrusion->orientation.setNum(numw+1);
  SbRotation *A = extrusion->orientation.startEditing();
  for(int i=0; i<=numw; i++)
    A[i] = SbRotation(SbVec3f(0,1,0),-numw*dphi/2+i*dphi);
  extrusion->orientation.finishEditing();
  extrusion->orientation.setDefault(FALSE);

 // set pitch angle
  extrusion->scale.setNum(numw+1);
  SbVec2f *sc = extrusion->scale.startEditing();
  for(int i=0; i<=numw; i++) {
    double scale = 1-tan(ga)*(i*dw-width/2)/ra;
    sc[i] = SbVec2f(scale,scale); // first x-scale / second z-scale
  }
  extrusion->scale.finishEditing();
  extrusion->scale.setDefault(FALSE);

  // additional flags
//  extrusion->solid=TRUE; // backface culling
  extrusion->convex=FALSE; // only convex polygons included in visualisation
//  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
//  extrusion->beginCap=TRUE; // front side at begin of the spine
//  extrusion->endCap=TRUE; // front side at end of the spine
  extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth
}
 
void GearWheel::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(5, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getNumberOfTeeth, &OpenMBV::GearWheel::setNumberOfTeeth);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getWidth, &OpenMBV::GearWheel::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getHelixAngle, &OpenMBV::GearWheel::setHelixAngle);
    FloatEditor *pitchAngleEditor=new FloatEditor(properties, QIcon(), "Pitch angle");
    pitchAngleEditor->setRange(-M_PI/4, M_PI/4);
    pitchAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getPitchAngle, &OpenMBV::GearWheel::setPitchAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getModule, &OpenMBV::GearWheel::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure Angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getPressureAngle, &OpenMBV::GearWheel::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getBacklash, &OpenMBV::GearWheel::setBacklash);
  }
}

}
