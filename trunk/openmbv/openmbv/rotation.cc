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
#include "openmbvcppinterface/rotation.h"
#include <QMenu>

using namespace std;

Rotation::Rotation(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  OpenMBV::Rotation* rot=(OpenMBV::Rotation*)obj;
  iconFile="rotation.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  // read XML
  vector<OpenMBV::PolygonPoint*>* contour=rot->getContour();

  // create so
  int open=fabs(rot->getEndAngle()-rot->getStartAngle()-2*M_PI)<1e-6?0:1;
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
    n1->vector.set1Value(0, sin(rot->getStartAngle()), 0, -cos(rot->getStartAngle()));
    // face
    csf1=new IndexedTesselationFace;
    soSepRigidBody->addChild(csf1);
    csf1->windingRule.setValue(IndexedTesselationFace::ODD);
    csf1->coordinate.connectFrom(&v->point);
    // normal
    SoNormal *n2=new SoNormal;
    soSepRigidBody->addChild(n2);
    n2->vector.set1Value(0, -sin(rot->getEndAngle()), 0, cos(rot->getEndAngle()));
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
  unsigned int cs=contour?contour->size():0;
  const unsigned int rs=(unsigned int)(20/2/M_PI*(rot->getEndAngle()-rot->getStartAngle()))+open;
  int nrv=0, nrf=0, nrn=0, nrl=0, nrcsf=0, nrcsl=0;
  for(unsigned int c=0; c<cs; c++) {
    unsigned int cn=c+1; if(cn>=cs) cn=0;
    unsigned int cp; if(c==0) cp=cs-1; else cp=c-1;
    for(unsigned int r=0; r<rs; r++) {
      unsigned int rn=r+1; if(rn>=rs) rn=0;
      double a=rot->getStartAngle()+(rot->getEndAngle()-rot->getStartAngle())/(rs-open)*r;
      // coord
      v->point.set1Value(nrv++, (*contour)[c]->getXComponent()*cos(a),(*contour)[c]->getYComponent(),(*contour)[c]->getXComponent()*sin(a));
      // normal
      SbVec2f np((*contour)[c]->getYComponent()-(*contour)[cp]->getYComponent(),(*contour)[cp]->getXComponent()-(*contour)[c]->getXComponent()); np.normalize(); //x-y-plane
      SbVec2f nn((*contour)[cn]->getYComponent()-(*contour)[c]->getYComponent(),(*contour)[c]->getXComponent()-(*contour)[cn]->getXComponent()); nn.normalize(); //x-y-plane
      if(((int)((*contour)[c]->getBorderValue()+0.5))!=1)
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
      if(((int)((*contour)[c]->getBorderValue()+0.5))==1)
        l->coordIndex.set1Value(nrl++, rs*c+r);
    }
    // line
    if(((int)((*contour)[c]->getBorderValue()+0.5))==1) {
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

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    FloatEditor *startAngleEditor=new FloatEditor(properties, QIcon(), "Start angle");
    startAngleEditor->setRange(0, 360); // degree
    startAngleEditor->setStep(10); // degree
    startAngleEditor->setSuffix(QString::fromUtf8("\xc2\xb0")); // utf8 degree sign
    startAngleEditor->setFactor(M_PI/180); // degree to rad conversion factor
    startAngleEditor->setOpenMBVParameter(rot, &OpenMBV::Rotation::getStartAngle, &OpenMBV::Rotation::setStartAngle);

    FloatEditor *endAngleEditor=new FloatEditor(properties, QIcon(), "End angle");
    endAngleEditor->setRange(0, 360); // degree
    endAngleEditor->setStep(10); // degree
    endAngleEditor->setSuffix(QString::fromUtf8("\xc2\xb0")); // utf8 degree sign
    endAngleEditor->setFactor(M_PI/180); // degree to rad conversion factor
    endAngleEditor->setOpenMBVParameter(rot, &OpenMBV::Rotation::getEndAngle, &OpenMBV::Rotation::setEndAngle);

    FloatMatrixEditor *contourEditor=new FloatMatrixEditor(properties, QIcon(), "Contour", 0, 3);
    contourEditor->setOpenMBVParameter(rot, &OpenMBV::Rotation::getContour, &OpenMBV::Rotation::setContour);
  }
}
