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
#include "dynamicindexedlineset.h"
#include "utils.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include "openmbvcppinterface/dynamicindexedlineset.h"
#include <QMenu>
#include <vector>
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

DynamicIndexedLineSet::DynamicIndexedLineSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : FlexibleBody(obj, parentItem, soParent, ind) {
  lineset=std::static_pointer_cast<OpenMBV::DynamicIndexedLineSet>(obj);
  iconFile="indexedlineset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  auto *line = new SoIndexedLineSet;
  line->coordIndex.setValues(0, lineset->getIndices().size(), lineset->getIndices().data());
  line->materialIndex.setValues(0, lineset->getIndices().size(), lineset->getIndices().data());
  soSep->addChild(line);
}

double DynamicIndexedLineSet::update() {
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = lineset->getRow(frame);

  SbColor *colorData = mat->diffuseColor.startEditing();
  SbColor *specData = mat->specularColor.startEditing();
  float h, s, v;
  mat->diffuseColor[0].getHSVValue(h, s, v);
  double m=1/(maximalColorValue-minimalColorValue);

  points->point.setNum(lineset->getNumberOfVertexPositions());
  SbVec3f *pointData = points->point.startEditing();
  for (int i=0; i<lineset->getNumberOfVertexPositions(); i++) {
    double hue = diffuseColor[0];
    if(hue<0) {
    double col = data[i*4+4];
    col = m*(col-minimalColorValue);
    if(col<0) col=0;
    if(col>1) col=1;
    hue = (1-col)*2/3;
    }
    colorData[i].setHSVValue(hue, s, v);
    specData[i].setHSVValue(hue, 0.7*s, v);
    pointData[i][0] = data[i*4+1];
    pointData[i][1] = data[i*4+2];
    pointData[i][2] = data[i*4+3];
  }
  mat->diffuseColor.finishEditing();
  mat->diffuseColor.setDefault(FALSE);
  mat->specularColor.finishEditing();
  mat->specularColor.setDefault(FALSE);
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
