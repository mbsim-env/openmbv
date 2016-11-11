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
#include "dynamicindexedfaceset.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/dynamicindexedfaceset.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

DynamicIndexedFaceSet::DynamicIndexedFaceSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind) {
  faceset=std::static_pointer_cast<OpenMBV::DynamicIndexedFaceSet>(obj);
  iconFile="indexedfaceset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  vector<int> indices = faceset->getIndices();
  int idx[indices.size()];
  for(unsigned int i=0; i<indices.size(); i++)
    idx[i] = indices[i];

  points = new SoCoordinate3;
  SoIndexedFaceSet *surface = new SoIndexedFaceSet;
  surface->coordIndex.setValues(0, indices.size(), idx);
  soSep->addChild(points);
  soSep->addChild(surface);

  // outline
  soSep->addChild(soOutLineSwitch);
}

void DynamicIndexedFaceSet::createProperties() {
  DynamicColoredBody::createProperties();
}

double DynamicIndexedFaceSet::update() {
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = faceset->getRow(frame);

  points->point.setNum(faceset->getNumberOfVertexPositions());
  SbVec3f *pointData = points->point.startEditing();
  for (int i=0; i<faceset->getNumberOfVertexPositions(); i++) {
    pointData[i][0] = data[i*3+1];
    pointData[i][1] = data[i*3+2];
    pointData[i][2] = data[i*3+3];
  }
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
