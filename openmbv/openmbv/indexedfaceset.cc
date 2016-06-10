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
#include "indexedfaceset.h"
#include "utils.h"
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/indexedfaceset.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

IndexedFaceSet::IndexedFaceSet(const boost::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : RigidBody(obj, parentItem, soParent, ind) {
  faceset=boost::static_pointer_cast<OpenMBV::IndexedFaceSet>(obj);
  iconFile="indexedfaceset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<vector<double> > vp = faceset->getVertexPositions();
  vector<int> indices = faceset->getIndices();

   float pts[vp.size()][3];
   for(unsigned int i=0; i<vp.size(); i++) {
     for(unsigned int j=0; j<3; j++)
       pts[i][j] = vp[i][j];
   }
   int idx[indices.size()];
   for(unsigned int i=0; i<indices.size(); i++)
     idx[i] = indices[i];

   SoCoordinate3 *points = new SoCoordinate3;
   SoIndexedFaceSet *surface = new SoIndexedFaceSet;
   points->point.setValues(0, vp.size(), pts);
   surface->coordIndex.setValues(0, indices.size(), idx);
   soSepRigidBody->addChild(points);
   soSepRigidBody->addChild(surface);

   // outline
   soSepRigidBody->addChild(soOutLineSwitch);
}

void IndexedFaceSet::createProperties() {
  RigidBody::createProperties();
}

}
