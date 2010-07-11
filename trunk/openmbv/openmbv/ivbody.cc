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
#include "utils.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/group.h"

using namespace std;

IvBody::IvBody(OpenMBV::Object *obj, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(obj, h5Parent, parentItem, soParent) {
  OpenMBV::IvBody *ivb=(OpenMBV::IvBody*)obj;
  iconFile=":/ivbody.svg";
  setIcon(0, Utils::QIconCached(iconFile.c_str()));

  // read XML
  string fileName=ivb->getIvFileName();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=OpenMBV::Object::fixPath(ivb->getSeparateGroup()->getFileName(), fileName);

  // create so
  SoGroup *soIv=Utils::SoDBreadAllCached(fileName.c_str());
  soSepRigidBody->addChild(soIv);
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
  double size=min(x2-x1,min(y2-y1,z2-z1));
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);

  // outline
  if(ivb->getCreaseEdges()>=0 || ivb->getBoundaryEdges()) {
    Utils::Edges *edges=NULL;
    soSepRigidBody->addChild(soOutLineSwitch);
    soOutLineSep->addChild(Utils::preCalculateEdgesCached(soIv, edges));
    if(ivb->getCreaseEdges()>=0) soOutLineSep->addChild(Utils::calculateCreaseEdges(ivb->getCreaseEdges(), edges));
    if(ivb->getBoundaryEdges()) soOutLineSep->addChild(Utils::calculateBoundaryEdges(edges));
  }
}
