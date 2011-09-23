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
#include "ivbody.h"
#include <Inventor/nodes/SoFile.h>
#include <Inventor/nodes/SoDrawStyle.h>

#include <Inventor/fields/SoMFColor.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>

#include <vector>
#include "edgecalculation.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/group.h"

#include <QtConcurrentRun>

using namespace std;

IvBody::IvBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind), calculateEdgesThread(this) {
  OpenMBV::IvBody *ivb=(OpenMBV::IvBody*)obj;
  iconFile=":/ivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));
  edgeCalc=NULL;

  // read XML
  string fileName=ivb->getIvFileName();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=OpenMBV::Object::fixPath(ivb->getSeparateGroup()->getFileName(), fileName);

  // create so
  SoSeparator *sep=new SoSeparator; // to enable caching
  soSepRigidBody->addChild(sep);
  SoGroup *soIv=Utils::SoDBreadAllCached(fileName.c_str());
  sep->addChild(soIv);
  // connect object OpenMBVIvBodyMaterial in file to hdf5 mat if it is of type SoMaterial
  SoBase *ref=SoNode::getByName("OpenMBVIvBodyMaterial");
  if(ref && ref->getTypeId()==SoMaterial::getClassTypeId()) {
    ((SoMaterial*)ref)->diffuseColor.connectFrom(&mat->diffuseColor);
    ((SoMaterial*)ref)->specularColor.connectFrom(&mat->specularColor);
  }

  // scale ref/localFrame
  SoGetBoundingBoxAction bboxAction(SbViewportRegion(0,0));
  bboxAction.apply(soSepRigidBody);
  float x1,y1,z1,x2,y2,z2;
  bboxAction.getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
  double size=min(x2-x1,min(y2-y1,z2-z1))*ivb->getScaleFactor();
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  if(ivb->getCreaseEdges()>=0 || ivb->getBoundaryEdges()) {
    soSepRigidBody->addChild(soOutLineSwitch);
    // get data for edge calculation from scene
    edgeCalc=new EdgeCalculation(soIv);
    // pre calculate edges, calculate crease edges and boundary edges in thread and call addEdgesToScene if finished
    connect(&calculateEdgesThread, SIGNAL(finished()), this, SLOT(addEdgesToScene()));
    calculateEdgesThread.start(QThread::IdlePriority);
  }
}

IvBody::~IvBody() {
  delete edgeCalc;
}

void IvBody::calculateEdges() {
  OpenMBV::IvBody *ivb=(OpenMBV::IvBody*)object;
  cout<<"Started edge calculation for "<<ivb->getFullName()<<" in a thread: ";
  edgeCalc->preproces(true);
  if(ivb->getCreaseEdges()>=0) edgeCalc->calcCreaseEdges(ivb->getCreaseEdges());
  if(ivb->getBoundaryEdges()) edgeCalc->calcBoundaryEdges();
}

void IvBody::addEdgesToScene() {
  OpenMBV::IvBody *ivb=(OpenMBV::IvBody*)object;
  soOutLineSep->addChild(edgeCalc->getCoordinates());
  if(ivb->getCreaseEdges()>=0) soOutLineSep->addChild(edgeCalc->getCreaseEdges());
  if(ivb->getBoundaryEdges()) soOutLineSep->addChild(edgeCalc->getBoundaryEdges());
  cout<<"Finished edge calculation for "<<ivb->getFullName()<<" and added to scene."<<endl;
}
