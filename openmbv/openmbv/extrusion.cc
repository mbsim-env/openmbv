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
#include "extrusion.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <vector>
#include "utils.h"
#include "openmbvcppinterface/extrusion.h"
#include <QMenu>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

Extrusion::Extrusion(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  e=std::static_pointer_cast<OpenMBV::Extrusion>(obj);
  iconFile="extrusion.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  OpenMBV::Extrusion::WindingRule windingRule_=e->getWindingRule();
  int windingRule=GLU_TESS_WINDING_ODD;
  if(windingRule_==OpenMBV::Extrusion::odd) windingRule=GLU_TESS_WINDING_ODD; 
  if(windingRule_==OpenMBV::Extrusion::nonzero) windingRule=GLU_TESS_WINDING_NONZERO;
  if(windingRule_==OpenMBV::Extrusion::positive) windingRule=GLU_TESS_WINDING_POSITIVE;
  if(windingRule_==OpenMBV::Extrusion::negative) windingRule=GLU_TESS_WINDING_NEGATIVE;
  if(windingRule_==OpenMBV::Extrusion::absGEqTwo) windingRule=GLU_TESS_WINDING_ABS_GEQ_TWO;
  double height=e->getHeight();
  const bool hasHeight=fabs(height)>1e-13;
  if(!hasHeight) height=0;
  std::vector<shared_ptr<std::vector<shared_ptr<OpenMBV::PolygonPoint> > > > contour=e->getContours();

  // create so
  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  // two side render if !hasHeight
  if(!hasHeight) {
    auto *sh=new SoShapeHints;
    soSepRigidBody->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  }
  auto *side=new SoSeparator;
  soSepRigidBody->addChild(side);
  if(hasHeight) {
    // side
    // shape hint
    auto *sh=new SoShapeHints;
    side->addChild(sh);
    sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
    sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
    sh->creaseAngle.setValue(M_PI);
  }
  // side
  for(auto & c : contour) {
    SoNormal *n=nullptr;
    if(hasHeight) {
      n=new SoNormal;
      side->addChild(n);
    }
    auto *v=new SoCoordinate3;
    side->addChild(v);
    size_t r;
    SoIndexedFaceSet *s=nullptr;
    if(hasHeight) {
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
    for(r=0; r<c->size(); r++) {
      size_t rn=r+1; if(rn>=c->size()) rn=0;
      size_t rp; if(r>=1) rp=r-1; else rp=c->size()-1;
      v->point.set1Value(2*r+0, (*c)[r]->getXComponent(), (*c)[r]->getYComponent(), 0);
      v->point.set1Value(2*r+1, (*c)[r]->getXComponent(), (*c)[r]->getYComponent(), height);
      if(hasHeight) {
        SbVec3f n1((*c)[r]->getYComponent()-(*c)[rp]->getYComponent(),(*c)[rp]->getXComponent()-(*c)[r]->getXComponent(),0); n1.normalize();
        SbVec3f n2((*c)[rn]->getYComponent()-(*c)[r]->getYComponent(),(*c)[r]->getXComponent()-(*c)[rn]->getXComponent(),0); n2.normalize();
        if(((int)((*c)[r]->getBorderValue()+0.5))!=1)
          n1=n2=n1+n2;
        n->vector.set1Value(2*r+0, n1);
        n->vector.set1Value(2*r+1, n2);
      }
      ol1->coordIndex.set1Value(r, 2*r+0);
      ol2->coordIndex.set1Value(r, 2*r+1);
      if(((int)((*c)[r]->getBorderValue()+0.5))==1) {
        ol3->coordIndex.set1Value(nr++, 2*r+0);
        ol3->coordIndex.set1Value(nr++, 2*r+1);
        ol3->coordIndex.set1Value(nr++, -1);
      }
      if(hasHeight) {
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
  }
  // base and top
  gluTessProperty(Utils::tess, GLU_TESS_WINDING_RULE, windingRule);
  auto *soTess=new SoGroup;
  soTess->ref();
  vector<GLdouble*> vPtr;
  vPtr.reserve(contour.size()*(!contour.empty()?contour[0]->size():0)*2);
  gluTessBeginPolygon(Utils::tess, soTess);
  for(auto & c : contour) {
    gluTessBeginContour(Utils::tess);
    for(size_t r=0; r<c->size(); r++) {
      auto *v=new GLdouble[3]; // is deleted later using vPtr
      vPtr.push_back(v);
      v[0]=(*c)[r]->getXComponent();
      v[1]=(*c)[r]->getYComponent();
      v[2]=0;
      gluTessVertex(Utils::tess, v, v);
    }
    gluTessEndContour(Utils::tess);
  }
  gluTessEndPolygon(Utils::tess);
  // now we can delete all v
  for(auto & i : vPtr)
    delete[]i;
  // normal binding
  auto *nb=new SoNormalBinding;
  soSepRigidBody->addChild(nb);
  nb->value.setValue(SoNormalBinding::OVERALL);
  // normal
  auto *n=new SoNormal;
  soSepRigidBody->addChild(n);
  n->vector.set1Value(0, 0, 0, -1);
  // vertex ordering
  auto *sh=new SoShapeHints;
  soSepRigidBody->addChild(sh);
  sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  // base
  soSepRigidBody->addChild(soTess);
  soTess->unref();
  if(hasHeight) {
    // trans
    auto *t=new SoTranslation;
    soSepRigidBody->addChild(t);
    t->translation.setValue(0, 0, height);
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
 
void Extrusion::createProperties() {
  RigidBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    auto *windingRuleEditor=new ComboBoxEditor(properties, QIcon(), "Winding rule", {
      make_tuple(OpenMBV::Extrusion::odd,        "Odd",             QIcon(), "Extrusion::windingRule::odd"),
      make_tuple(OpenMBV::Extrusion::nonzero,    "Nonzero",         QIcon(), "Extrusion::windingRule::nonzero"),
      make_tuple(OpenMBV::Extrusion::positive,   "Positive",        QIcon(), "Extrusion::windingRule::positive"),
      make_tuple(OpenMBV::Extrusion::negative,   "Negative",        QIcon(), "Extrusion::windingRule::negative"),
      make_tuple(OpenMBV::Extrusion::absGEqTwo,  "Abs. value >= 2", QIcon(), "Extrusion::windingRule::absgt2")
    });
    windingRuleEditor->setOpenMBVParameter(e, &OpenMBV::Extrusion::getWindingRule, &OpenMBV::Extrusion::setWindingRule);

    auto *heightEditor=new FloatEditor(properties, QIcon(), "Height");
    heightEditor->setRange(0, DBL_MAX);
    heightEditor->setOpenMBVParameter(e, &OpenMBV::Extrusion::getHeight, &OpenMBV::Extrusion::setHeight);

    new NotAvailableEditor(properties, QIcon(), "Contours");
  }
}

}
