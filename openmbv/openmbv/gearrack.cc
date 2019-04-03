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
#include "gearrack.h"
#include <Inventor/VRMLnodes/SoVRMLExtrusion.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/gearrack.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

GearRack::GearRack(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::GearRack>(obj);
  iconFile="gearrack.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  int z = e->getNumberOfTeeth();
  double h = e->getHeight();
  double width = e->getWidth();
  double be = e->getHelixAngle();
  double al0 = atan(tan(e->getPressureAngle())/cos(be));
  double m = e->getModule()/cos(be);
  double b = e->getBacklash();

  double d0 = m*z;
  double p0 = M_PI*m;
  double c = 0.167*m;
  double s0 = p0/2;
  double df = d0 - 2*(m+c);
  double da = d0 + 2*m;
  double db = d0*cos(al0);
  double phi0 = tan(al0) - al0;
  double ala = acos(db/da);
  double phia = tan(ala) - ala;
  double sb = p0/2 + 2*m*tan(al0) - b;
  vector<double> x(7), y(7);
  x[0] = -p0/2;
  y[0] = -m-c;
  x[1] = -sb/2;
  y[1] = y[0];
  x[2] = x[1];
  y[2] = -m;
  x[3] = x[2]+2*m*tan(al0);
  y[3] = m;
  for (int i=6, j=1; i>=4; i--,j++) {
    x[i] = -x[j];
    y[i] = y[j];
  }
  vector<double> X(z*x.size()+3);
  vector<double> Y(X.size()+3);
  for(int i=0; i<z; i++) {
    int k = i*x.size();
    for(unsigned int j=0; j<x.size(); j++) {
      X[k+j] = x[j]+i*p0;
      Y[k+j] = y[j];
    }
  }
  X[X.size()-3] = X[X.size()-4]+(p0-sb)/2;
  Y[X.size()-3] = Y[X.size()-4];
  X[X.size()-2] = X[X.size()-3];
  Y[X.size()-2] = Y[X.size()-3]-(h-2*m);
  X[X.size()-1] = X[0];
  Y[X.size()-1] = Y[X.size()-2];

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
  int numw = 2;
  double dw = width/numw;
//  double dx = dw*tan(be);
  extrusion->spine.setNum(numw+1);
  SbVec3f *sp = extrusion->spine.startEditing();
  for(int i=0; i<=numw; i++)
    sp[i] = SbVec3f(0,i*dw,0);
//    sp[i] = SbVec3f(-numw*dx/2+i*dx,i*dw,0);
  extrusion->spine.finishEditing();
  extrusion->spine.setDefault(FALSE);

  // additional flags
  //  extrusion->solid=TRUE; // backface culling
  extrusion->convex=FALSE; // only convex polygons included in visualisation
  //  extrusion->ccw=TRUE; // vertex ordering counterclockwise?
  //  extrusion->beginCap=TRUE; // front side at begin of the spine
  //  extrusion->endCap=TRUE; // front side at end of the spine
  extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth
}
 
void GearRack::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    IntEditor *numEditor=new IntEditor(properties, QIcon(), "Number of teeth");
    numEditor->setRange(1, 100);
    numEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getNumberOfTeeth, &OpenMBV::GearRack::setNumberOfTeeth);
    FloatEditor *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getHeight, &OpenMBV::GearRack::setHeight);
    FloatEditor *widthEditor=new FloatEditor(properties, QIcon(), "Width");
    widthEditor->setRange(0, DBL_MAX);
    widthEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getWidth, &OpenMBV::GearRack::setWidth);
    FloatEditor *helixAngleEditor=new FloatEditor(properties, QIcon(), "Helix angle");
    helixAngleEditor->setRange(-M_PI/4, M_PI/4);
    helixAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getHelixAngle, &OpenMBV::GearRack::setHelixAngle);
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getModule, &OpenMBV::GearRack::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getPressureAngle, &OpenMBV::GearRack::setPressureAngle);
    FloatEditor *backlashEditor=new FloatEditor(properties, QIcon(), "Backlash");
    backlashEditor->setRange(0, 0.005);
    backlashEditor->setOpenMBVParameter(e, &OpenMBV::GearRack::getBacklash, &OpenMBV::GearRack::setBacklash);
  }
}

}
