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
#include "rotation.h"
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedLineSet.h>

using namespace std;

Rotation::Rotation(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/rotation.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // read XML
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"startAngle");
  float startAngle=0;
  if(e)
    startAngle=Utils::toDouble(e->GetText());
  e=element->FirstChildElement(OPENMBVNS"endAngle");
  float endAngle=2*M_PI;
  if(e)
    endAngle=Utils::toDouble(e->GetText());
  e=element->FirstChildElement(OPENMBVNS"contour");
  vector<vector<double> > contour;
  contour=Utils::toMatrix(e->GetText());

  // create so
  int open=fabs(endAngle-startAngle-2*M_PI)<1e-6?0:1;
  // coord, normal, face
  SoCoordinate3 *v=new SoCoordinate3;
  soSepRigidBody->addChild(v);
  SoNormal *n=new SoNormal;
  soSepRigidBody->addChild(n);
  SoIndexedFaceSet *f=new SoIndexedFaceSet;
  soSepRigidBody->addChild(f);
  // outline
  soSepRigidBody->addChild(soOutLineSwitch);
  SoIndexedLineSet *l=new SoIndexedLineSet;
  soOutLineSep->addChild(l);
  // cross sections
  IndexedTesselationFace *csf1=NULL, *csf2=NULL;
  SoIndexedLineSet *csl1=NULL, *csl2=NULL;
  if(open) {
    // normal binding
    SoNormalBinding *nb=new SoNormalBinding;
    soSepRigidBody->addChild(nb);
    nb->value.setValue(SoNormalBinding::OVERALL);
    // normal
    SoNormal *n1=new SoNormal;
    soSepRigidBody->addChild(n1);
    n1->vector.set1Value(0, sin(startAngle), 0, -cos(startAngle));
    // face
    csf1=new IndexedTesselationFace;
    soSepRigidBody->addChild(csf1);
    csf1->windingRule.setValue(IndexedTesselationFace::ODD);
    csf1->coordinate.connectFrom(&v->point);
    // normal
    SoNormal *n2=new SoNormal;
    soSepRigidBody->addChild(n2);
    n2->vector.set1Value(0, -sin(endAngle), 0, cos(endAngle));
    // face
    csf2=new IndexedTesselationFace;
    soSepRigidBody->addChild(csf2);
    csf2->windingRule.setValue(IndexedTesselationFace::ODD);
    csf2->coordinate.connectFrom(&v->point);
    // outline
    csl1=new SoIndexedLineSet;
    soOutLineSep->addChild(csl1);
    csl2=new SoIndexedLineSet;
    soOutLineSep->addChild(csl2);
  }
  // coord, normal, face
  unsigned int cs=contour.size();
  const unsigned int rs=(unsigned int)(20/2/M_PI*(endAngle-startAngle))+open;
  int nrv=0, nrf=0, nrn=0, nrl=0, nrcsf=0, nrcsl=0;
  for(unsigned int c=0; c<cs; c++) {
    unsigned int cn=c+1; if(cn>=cs) cn=0;
    unsigned int cp; if(c==0) cp=cs-1; else cp=c-1;
    for(unsigned int r=0; r<rs; r++) {
      unsigned int rn=r+1; if(rn>=rs) rn=0;
      double a=startAngle+(endAngle-startAngle)/(rs-open)*r;
      // coord
      v->point.set1Value(nrv++, contour[c][0]*cos(a),contour[c][1],contour[c][0]*sin(a));
      // normal
      SbVec2f np(contour[c][1]-contour[cp][1],contour[cp][0]-contour[c][0]); np.normalize(); //x-y-plane
      SbVec2f nn(contour[cn][1]-contour[c][1],contour[c][0]-contour[cn][0]); nn.normalize(); //x-y-plane
      if(((int)(contour[c][2]+0.5))!=1)
        nn=np=nn+np;
      n->vector.set1Value(nrn++, np[0]*cos(a),np[1],np[0]*sin(a));
      n->vector.set1Value(nrn++, nn[0]*cos(a),nn[1],nn[0]*sin(a));
      // face
      if(r<rs-open) {
        f->coordIndex.set1Value(nrf,    rs*c+r);
        f->normalIndex.set1Value(nrf++, 2*(rs*c+r)+1);
        f->coordIndex.set1Value(nrf,    rs*cn+r);
        f->normalIndex.set1Value(nrf++, 2*(rs*cn+r)+0);
        f->coordIndex.set1Value(nrf,    rs*cn+rn);
        f->normalIndex.set1Value(nrf++, 2*(rs*cn+rn)+0);
        f->coordIndex.set1Value(nrf,    rs*c+rn);
        f->normalIndex.set1Value(nrf++, 2*(rs*c+rn)+1);
        f->coordIndex.set1Value(nrf,    -1);
        f->normalIndex.set1Value(nrf++, -1);
      }
      // line
      if(((int)(contour[c][2]+0.5))==1)
        l->coordIndex.set1Value(nrl++, rs*c+r);
    }
    // line
    if(((int)(contour[c][2]+0.5))==1) {
      if(!open)
        l->coordIndex.set1Value(nrl++, rs*c+0);
      l->coordIndex.set1Value(nrl++, -1);
    }
    if(open) {
      csf1->coordIndex.set1Value(nrcsf, rs*c+0);
      csf2->coordIndex.set1Value(nrcsf++, rs*c+(rs-1));
      csl1->coordIndex.set1Value(nrcsl, rs*c+0);
      csl2->coordIndex.set1Value(nrcsl++, rs*c+(rs-1));
    }
  }
  if(open) {
    csf1->coordIndex.set1Value(nrcsf, -1);
    csf2->coordIndex.set1Value(nrcsf++, -1);
    csf1->generate();
    csf2->generate();
    csl1->coordIndex.set1Value(nrcsl, rs*0+0);
    csl2->coordIndex.set1Value(nrcsl++, rs*0+(rs-1));
    csl1->coordIndex.set1Value(nrcsl, -1);
    csl2->coordIndex.set1Value(nrcsl++, -1);
  }
}
