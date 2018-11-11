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
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
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
  double width = e->getWidth();
  int z = e->getNumberOfTeeth();
  double al0 = e->getPressureAngle(); 
  double m = e->getModule();
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
  double sb = db*(s0/d0+phi0);
  double deb = 2*sb/db;
  double ga0 = 2*p0/d0;
  double rb = db/2;
  double ra = da/2;
  double rf = df/2;
  int numb = 5;
  double dphi = (ga0-deb)/(2*numb);
  double phi = ga0/2;
  vector<double> x(6*numb), y(6*numb);
  for (int i=0; i<numb; i++) {
    x[i] = rf*sin(phi);
    y[i] = rf*cos(phi);
    phi -= dphi;
  }
  dphi = (phia+ala)/numb;
  phi = 0;
  for (int i=numb; i<2*numb; i++) {
    x[i] = rb*(cos(deb/2)*(-sin(phi)+cos(phi)*phi) + sin(deb/2)*(cos(phi)+sin(phi)*phi));
    y[i] = rb*(-sin(deb/2)*(-sin(phi)+cos(phi)*phi) + cos(deb/2)*(cos(phi)+sin(phi)*phi));
    phi += dphi;
  }
  dphi = (deb/2 - phia)/numb;
  phi = deb/2 - phia;
  for (int i=2*numb; i<3*numb; i++) {
    x[i] = ra*sin(phi);
    y[i] = ra*cos(phi);
    phi -= dphi;
  }
  for (int i=6*numb-1, j=0; i>=3*numb; i--,j++) {
    x[i] = -x[j];
    y[i] = y[j];
  }
  vector<double> X(z*x.size());
  vector<double> Y(z*x.size());
  for(int i=0; i<z; i++) {
    int k = i*x.size();
    for(unsigned int j=0; j<x.size(); j++) {
      X[k+j] = cos(i*ga0)*x[j] - sin(i*ga0)*y[j];
      Y[k+j] = sin(i*ga0)*x[j] + cos(i*ga0)*y[j];
    }
  }

  int windingRule = GLU_TESS_WINDING_ODD;
  const bool hasWidth = fabs(width)>1e-13;
  if(!hasWidth) width = 0;
  else {
    auto *t=new SoTranslation;
    soSepRigidBody->addChild(t);
    t->translation.setValue(0, 0, -width/2);
  }
  // create so
  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  // two side render if !hasWidth
  if(!hasWidth) {
    auto *sh=new SoShapeHints;
    soSepRigidBody->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  }
  auto *side=new SoSeparator;
  soSepRigidBody->addChild(side);
  if(hasWidth) {
    // side
    // shape hint
    auto *sh=new SoShapeHints;
    side->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
    sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
    sh->creaseAngle.setValue(M_PI);
  }
  // side
    SoNormal *n=nullptr;
    if(hasWidth) {
      n=new SoNormal;
      side->addChild(n);
    }
    auto *v=new SoCoordinate3;
    side->addChild(v);
    size_t r;
    SoIndexedFaceSet *s=nullptr;
    if(hasWidth) {
      s=new SoIndexedFaceSet;
      side->addChild(s);
    }
    // outline
    auto *ol1=new SoIndexedLineSet;
    auto *ol2=new SoIndexedLineSet;
    auto *ol3=new SoIndexedLineSet;
    soOutLineSep->addChild(v);
    soOutLineSep->addChild(ol1);
    soOutLineSep->addChild(ol2);
    int nr=0;
    soOutLineSep->addChild(ol3);
    //
    for(r=0; r<X.size(); r++) {
      size_t rn=r+1; if(rn>=X.size()) rn=0;
      size_t rp; if(r>=1) rp=r-1; else rp=X.size()-1;
      v->point.set1Value(2*r+0, X[r], Y[r], 0);
      v->point.set1Value(2*r+1, X[r], Y[r], width);
      if(hasWidth) {
        SbVec3f n1(Y[r]-Y[rp],X[rp]-X[r],0); n1.normalize();
        SbVec3f n2(Y[rn]-Y[r],X[r]-X[rn],0); n2.normalize();
        if(((int)(0+0.5))!=1)
          n1=n2=n1+n2;
        n->vector.set1Value(2*r+0, n1);
        n->vector.set1Value(2*r+1, n2);
      }
      ol1->coordIndex.set1Value(r, 2*r+0);
      ol2->coordIndex.set1Value(r, 2*r+1);
      if(((int)(0+0.5))==1) {
        ol3->coordIndex.set1Value(nr++, 2*r+0);
        ol3->coordIndex.set1Value(nr++, 2*r+1);
        ol3->coordIndex.set1Value(nr++, -1);
      }
      if(hasWidth) {
        s->coordIndex.set1Value(5*r+0, 2*r+0);
        s->coordIndex.set1Value(5*r+1, 2*r+1);
        s->coordIndex.set1Value(5*r+2, 2*rn+1);
        s->coordIndex.set1Value(5*r+3, 2*rn+0);
        s->coordIndex.set1Value(5*r+4, -1);
        s->normalIndex.set1Value(5*r+0, 2*r+1);
        s->normalIndex.set1Value(5*r+1, 2*r+1);
        s->normalIndex.set1Value(5*r+2, 2*rn);
        s->normalIndex.set1Value(5*r+3, 2*rn);
        s->normalIndex.set1Value(5*r+4, -1);
      }
    }
    ol1->coordIndex.set1Value(r, 0);
    ol2->coordIndex.set1Value(r, 1);
    ol1->coordIndex.set1Value(r+1, -1);
    ol2->coordIndex.set1Value(r+1, -1);

  // base and top
  gluTessProperty(Utils::tess, GLU_TESS_WINDING_RULE, windingRule);
  auto *soTess=new SoGroup;
  soTess->ref();
  vector<GLdouble*> vPtr;
  vPtr.reserve(X.size()*2);
  gluTessBeginPolygon(Utils::tess, soTess);
    gluTessBeginContour(Utils::tess);
    for(size_t r=0; r<X.size(); r++) {
      auto *v=new GLdouble[3]; // is deleted later using vPtr
      vPtr.push_back(v);
      v[0]=X[r];
      v[1]=Y[r];
      v[2]=0;
      gluTessVertex(Utils::tess, v, v);
    }
    gluTessEndContour(Utils::tess);
  gluTessEndPolygon(Utils::tess);
  // now we can delete all v
  for(auto & i : vPtr)
    delete[]i;
  // normal binding
  auto *nb=new SoNormalBinding;
  soSepRigidBody->addChild(nb);
  nb->value.setValue(SoNormalBinding::OVERALL);
  // normal
  n=new SoNormal;
  soSepRigidBody->addChild(n);
  n->vector.set1Value(0, 0, 0, -1);
  // vertex ordering
  auto *sh=new SoShapeHints;
  soSepRigidBody->addChild(sh);
  sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  // base
  soSepRigidBody->addChild(soTess);
  soTess->unref();
  if(hasWidth) {
    // trans
    auto *t=new SoTranslation;
    soSepRigidBody->addChild(t);
    t->translation.setValue(0, 0, width);
    // normal
    n=new SoNormal;
    soSepRigidBody->addChild(n);
    n->vector.set1Value(0, 0, 0, 1);
    // vertex ordering
    auto *sh=new SoShapeHints;
    soSepRigidBody->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::COUNTERCLOCKWISE);
    // top
    soSepRigidBody->addChild(soTess);
  }
  // scale ref/localFrame
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
    FloatEditor *moduleEditor=new FloatEditor(properties, QIcon(), "Module");
    moduleEditor->setRange(0, DBL_MAX);
    moduleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getModule, &OpenMBV::GearWheel::setModule);
    FloatEditor *pressureAngleEditor=new FloatEditor(properties, QIcon(), "Pressure Angle");
    pressureAngleEditor->setRange(0, M_PI/4);
    pressureAngleEditor->setOpenMBVParameter(e, &OpenMBV::GearWheel::getPressureAngle, &OpenMBV::GearWheel::setPressureAngle);
  }
}

}
