/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include "config.h"
#include "dynamiccoloredbody.h"
#include "openmbvcppinterface/dynamiccoloredbody.h"
#include "utils.h"
#include <QMenu>

using namespace std;

namespace OpenMBVGUI {

DynamicColoredBody::DynamicColoredBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind, bool perVertexIndexed) : Body(obj, parentItem, soParent, ind), color(0), oldColor(std::numeric_limits<double>::quiet_NaN()) {
  dcb=std::static_pointer_cast<OpenMBV::DynamicColoredBody>(obj);
  // read XML
  minimalColorValue=dcb->getMinimalColorValue();
  maximalColorValue=dcb->getMaximalColorValue();

  // define a material and base color based on the xml data
  mat=new SoMaterial;
  soSep->addChild(mat);
  diffuseColor=dcb->getDiffuseColor();
  mat->diffuseColor.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, diffuseColor[1], diffuseColor[2]);
  mat->specularColor.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, 0.7*diffuseColor[1], diffuseColor[2]);
  mat->transparency.setValue(dcb->getTransparency());
  mat->shininess.setValue(0.9);
  if(perVertexIndexed) {
    auto *myMaterialBinding = new SoMaterialBinding;
    myMaterialBinding->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    soSep->addChild(myMaterialBinding);
  }
  else {
    baseColor=new SoBaseColor;
    soSep->addChild(baseColor);
    baseColor->rgb.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, diffuseColor[1], diffuseColor[2]);
  }
}

void DynamicColoredBody::createProperties() {
  Body::createProperties();

  // GUI
  if(!clone) {
    auto *minimalColorValueEditor=new FloatEditor(properties, QIcon(), "Minimal color value", false);
    minimalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMinimalColorValue, &OpenMBV::DynamicColoredBody::setMinimalColorValue);
    connect(minimalColorValueEditor, &FloatEditor::stateChanged, this, [this](double s){
      minimalColorValue=s;
    });

    auto *maximalColorValueEditor=new FloatEditor(properties, QIcon(), "Maximal color value", false);
    maximalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMaximalColorValue, &OpenMBV::DynamicColoredBody::setMaximalColorValue);
    connect(maximalColorValueEditor, &FloatEditor::stateChanged, this, [this](double s){
      maximalColorValue=s;
    });

    auto *diffuseColorValue=new ColorEditor(properties, QIcon(), "Diffuse color", true);
    diffuseColorValue->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getDiffuseColor, &OpenMBV::DynamicColoredBody::setDiffuseColor);

    auto *transparencyValueEditor=new FloatEditor(properties, QIcon(), "Transparency value", false);
    transparencyValueEditor->setRange(0, 1);
    transparencyValueEditor->setStep(0.1);
    transparencyValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getTransparency, &OpenMBV::DynamicColoredBody::setTransparency);
    connect(transparencyValueEditor, &FloatEditor::stateChanged, this, [this](double s){
      mat->transparency.setValue(s);
    });
  }
}

void DynamicColoredBody::setColor(double col) {
  if(oldColor!=col) {
    color=col;
    oldColor=col;
    double m=1/(maximalColorValue-minimalColorValue);
    col=m*col-m*minimalColorValue;
    if(col<0) col=0;
    if(col>1) col=1;
    setHueColor((1-col)*2/3);
  }
}

void DynamicColoredBody::setHueColor(double hue) {
  float h, s, v;
  baseColor->rgb[0].getHSVValue(h, s, v);
  baseColor->rgb.setHSVValue(hue, s, v);
  mat->diffuseColor[0].getHSVValue(h, s, v);
  mat->diffuseColor.setHSVValue(hue, s, v);
  mat->specularColor[0].getHSVValue(h, s, v);
  mat->specularColor.setHSVValue(hue, s, v);
}

QString DynamicColoredBody::getInfo() {
  return Body::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Color:</b> %1").arg(getColor());
}

}
