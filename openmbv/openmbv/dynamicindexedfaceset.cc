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

  points = new SoCoordinate3;
  SoIndexedFaceSet *surface = new SoIndexedFaceSet;
  surface->coordIndex.setValues(0, faceset->getIndices().size(), faceset->getIndices().data());
  surface->materialIndex.setValues(0, faceset->getIndices().size(), faceset->getIndices().data());
  SoMaterialBinding *myMaterialBinding = new SoMaterialBinding;
  myMaterialBinding->value = SoMaterialBinding::PER_VERTEX_INDEXED;
  myMaterials = new SoMaterial;
  soSep->addChild(myMaterials);
  soSep->addChild(myMaterialBinding);
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

  myMaterials->diffuseColor.setNum(faceset->getNumberOfVertexPositions());
  SbColor *colorData = myMaterials->diffuseColor.startEditing();
  float h, s, v;
  double hue;
  mat->diffuseColor[0].getHSVValue(h, s, v);
  double m=1/(maximalColorValue-minimalColorValue);
  for (int i=0; i<faceset->getNumberOfVertexPositions(); i++) {
    hue =  data[i*4+4];
    hue = m*(hue-minimalColorValue);
    if(hue<0) hue=0;
    if(hue>1) hue=1;
    colorData[i].setHSVValue((1-hue)*2/3, s, v);
  }
  myMaterials->diffuseColor.finishEditing();
  myMaterials->diffuseColor.setDefault(FALSE);

  points->point.setNum(faceset->getNumberOfVertexPositions());
  SbVec3f *pointData = points->point.startEditing();
  for (int i=0; i<faceset->getNumberOfVertexPositions(); i++) {
    pointData[i][0] = data[i*4+1];
    pointData[i][1] = data[i*4+2];
    pointData[i][2] = data[i*4+3];
  }
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
