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
#include "extrusion.h"
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <vector>

using namespace std;

Extrusion::Extrusion(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/extrusion.svg";
  setIcon(0, QIconCached(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"windingRule");
  string windingRule_=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
  int windingRule;
  if(windingRule_=="odd") windingRule=GLU_TESS_WINDING_ODD; 
  if(windingRule_=="nonZero") windingRule=GLU_TESS_WINDING_NONZERO;
  if(windingRule_=="positive") windingRule=GLU_TESS_WINDING_POSITIVE;
  if(windingRule_=="negative") windingRule=GLU_TESS_WINDING_NEGATIVE;
  if(windingRule_=="absGEqTwo") windingRule=GLU_TESS_WINDING_ABS_GEQ_TWO;
  e=e->NextSiblingElement();
  double height=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  vector<vector<vector<double> > > contour;
  while(e && e->ValueStr()==OPENMBVNS"contour") {
    contour.push_back(toMatrix(e->GetText()));
    e=e->NextSiblingElement();
  }

  // create so
  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  // side
  SoSeparator *side=new SoSeparator;
  soSepRigidBody->addChild(side);
  // shape hint
  SoShapeHints *sh=new SoShapeHints;
  side->addChild(sh);
  sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
  sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
  sh->creaseAngle.setValue(M_PI);
  // side
  for(size_t c=0; c<contour.size(); c++) {
    SoNormal *n=new SoNormal;
    side->addChild(n);
    SoCoordinate3 *v=new SoCoordinate3;
    side->addChild(v);
    size_t r;
    SoIndexedFaceSet *s=new SoIndexedFaceSet;
    side->addChild(s);
    // outline
    SoIndexedLineSet *ol1=new SoIndexedLineSet;
    SoIndexedLineSet *ol2=new SoIndexedLineSet;
    SoIndexedLineSet *ol3=new SoIndexedLineSet;
    soOutLineSep->addChild(v);
    soOutLineSep->addChild(ol1);
    soOutLineSep->addChild(ol2);
    int nr=0;
    soOutLineSep->addChild(ol3);
    //
    for(r=0; r<contour[c].size(); r++) {
      size_t rn=r+1; if(rn>=contour[c].size()) rn=0;
      size_t rp; if(r>=1) rp=r-1; else rp=contour[c].size()-1;
      v->point.set1Value(2*r+0, contour[c][r][0], contour[c][r][1], 0);
      v->point.set1Value(2*r+1, contour[c][r][0], contour[c][r][1], height);
      SbVec3f n1(contour[c][r][1]-contour[c][rp][1],contour[c][rp][0]-contour[c][r][0],0); n1.normalize();
      SbVec3f n2(contour[c][rn][1]-contour[c][r][1],contour[c][r][0]-contour[c][rn][0],0); n2.normalize();
      if(((int)(contour[c][r][2]+0.5))!=1)
        n1=n2=n1+n2;
      n->vector.set1Value(2*r+0, n1);
      n->vector.set1Value(2*r+1, n2);
      ol1->coordIndex.set1Value(r, 2*r+0);
      ol2->coordIndex.set1Value(r, 2*r+1);
      if(((int)(contour[c][r][2]+0.5))==1) {
        ol3->coordIndex.set1Value(nr++, 2*r+0);
        ol3->coordIndex.set1Value(nr++, 2*r+1);
        ol3->coordIndex.set1Value(nr++, -1);
      }
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
    ol1->coordIndex.set1Value(r, 0);
    ol2->coordIndex.set1Value(r, 1);
    ol1->coordIndex.set1Value(r+1, -1);
    ol2->coordIndex.set1Value(r+1, -1);
    // combine normals
    SoMFVec3f newvv;
    SoMFInt32 newvi;
    eps=1e-8;
    combine(n->vector, newvv, newvi);
    n->vector.copyFrom(newvv);
    convertIndex(s->normalIndex, newvi);
  }
  // base and top
  gluTessProperty(tess, GLU_TESS_WINDING_RULE, windingRule);
  SoGroup *soTess=new SoGroup;
  soTess->ref();
  gluTessBeginPolygon(tess, soTess);
  for(size_t c=0; c<contour.size(); c++) {
    gluTessBeginContour(tess);
    for(size_t r=0; r<contour[c].size(); r++) {
      GLdouble *v=new GLdouble[3];
      v[0]=contour[c][r][0];
      v[1]=contour[c][r][1];
      v[2]=0;
      gluTessVertex(tess, v, v);
    }
    gluTessEndContour(tess);
  }
  gluTessEndPolygon(tess);
  // normal binding
  SoNormalBinding *nb=new SoNormalBinding;
  soSepRigidBody->addChild(nb);
  nb->value.setValue(SoNormalBinding::OVERALL);
  // normal
  SoNormal *n=new SoNormal;
  soSepRigidBody->addChild(n);
  n->vector.set1Value(0, 0, 0, -1);
  // base
  soSepRigidBody->addChild(soTess);
  // trans
  SoTranslation *t=new SoTranslation;
  soSepRigidBody->addChild(t);
  t->translation.setValue(0, 0, height);
  // normal
  n=new SoNormal;
  soSepRigidBody->addChild(n);
  n->vector.set1Value(0, 0, 0, 1);
  // top
  soSepRigidBody->addChild(soTess);
  // scale ref/localFrame
}
