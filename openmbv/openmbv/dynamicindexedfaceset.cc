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

DynamicIndexedFaceSet::DynamicIndexedFaceSet(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  faceset=std::static_pointer_cast<OpenMBV::DynamicIndexedFaceSet>(obj);
  iconFile="indexedfaceset.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  minimalColorValue=faceset->getMinimalColorValue();
  maximalColorValue=faceset->getMaximalColorValue();

  points = new SoCoordinate3;
  SoIndexedFaceSet *surface = new SoIndexedFaceSet;
  surface->coordIndex.setValues(0, faceset->getIndices().size(), faceset->getIndices().data());
  surface->materialIndex.setValues(0, faceset->getIndices().size(), faceset->getIndices().data());
  SoMaterialBinding *myMaterialBinding = new SoMaterialBinding;
  myMaterialBinding->value = SoMaterialBinding::PER_VERTEX_INDEXED;
  myMaterials = new SoMaterial;
  std::vector<double> diffuseColor=faceset->getDiffuseColor();
  myMaterials->diffuseColor.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, diffuseColor[1], diffuseColor[2]);
  myMaterials->specularColor.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, 0.7*diffuseColor[1], diffuseColor[2]);
  myMaterials->transparency.setValue(faceset->getTransparency());
  myMaterials->shininess.setValue(0.9);
  soSep->addChild(myMaterials);
  soSep->addChild(myMaterialBinding);
  soSep->addChild(points);
  soSep->addChild(surface);

  // outline
  soSep->addChild(soOutLineSwitch);
}

void DynamicIndexedFaceSet::createProperties() {
  Body::createProperties();

  // GUI
  if(!clone) {
    FloatEditor *minimalColorValueEditor=new FloatEditor(properties, QIcon(), "Minimal color value");
    minimalColorValueEditor->setOpenMBVParameter(faceset, &OpenMBV::DynamicIndexedFaceSet::getMinimalColorValue, &OpenMBV::DynamicIndexedFaceSet::setMinimalColorValue);

    FloatEditor *maximalColorValueEditor=new FloatEditor(properties, QIcon(), "Maximal color value");
    maximalColorValueEditor->setOpenMBVParameter(faceset, &OpenMBV::DynamicIndexedFaceSet::getMaximalColorValue, &OpenMBV::DynamicIndexedFaceSet::setMaximalColorValue);

    ColorEditor *diffuseColorValue=new ColorEditor(properties, QIcon(), "Diffuse color", true);
    diffuseColorValue->setOpenMBVParameter(faceset, &OpenMBV::DynamicIndexedFaceSet::getDiffuseColor, &OpenMBV::DynamicIndexedFaceSet::setDiffuseColor);

    FloatEditor *transparencyValueEditor=new FloatEditor(properties, QIcon(), "Transparency value");
    transparencyValueEditor->setRange(0, 1);
    transparencyValueEditor->setStep(0.1);
    transparencyValueEditor->setOpenMBVParameter(faceset, &OpenMBV::DynamicIndexedFaceSet::getTransparency, &OpenMBV::DynamicIndexedFaceSet::setTransparency);
  }
}

double DynamicIndexedFaceSet::update() {
  int frame = MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data = faceset->getRow(frame);

  myMaterials->diffuseColor.setNum(faceset->getNumberOfVertexPositions());
  myMaterials->specularColor.setNum(faceset->getNumberOfVertexPositions());
  SbColor *colorData = myMaterials->diffuseColor.startEditing();
  SbColor *specData = myMaterials->specularColor.startEditing();
  float h, s, v;
  myMaterials->diffuseColor[0].getHSVValue(h, s, v);
  double m=1/(maximalColorValue-minimalColorValue);

  points->point.setNum(faceset->getNumberOfVertexPositions());
  SbVec3f *pointData = points->point.startEditing();
  for (int i=0; i<faceset->getNumberOfVertexPositions(); i++) {
    double col =  data[i*4+4];
    col = m*(col-minimalColorValue);
    if(col<0) col=0;
    if(col>1) col=1;
    double hue = (1-col)*2/3;
    colorData[i].setHSVValue(hue, s, v);
    specData[i].setHSVValue(hue, 0.7*s, v);
    pointData[i][0] = data[i*4+1];
    pointData[i][1] = data[i*4+2];
    pointData[i][2] = data[i*4+3];
  }
  myMaterials->diffuseColor.finishEditing();
  myMaterials->diffuseColor.setDefault(FALSE);
  myMaterials->specularColor.finishEditing();
  myMaterials->specularColor.setDefault(FALSE);
  points->point.finishEditing();
  points->point.setDefault(FALSE);
  return data[0]; //return time
}

}
