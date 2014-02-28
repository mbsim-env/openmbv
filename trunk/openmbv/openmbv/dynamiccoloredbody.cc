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
#include "dynamiccoloredbody.h"
#include "openmbvcppinterface/dynamiccoloredbody.h"
#include "utils.h"
#include <QMenu>

using namespace std;

namespace OpenMBVGUI {

DynamicColoredBody::DynamicColoredBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind), color(0), oldColor(nan("")) {
  OpenMBV::DynamicColoredBody *dcb=(OpenMBV::DynamicColoredBody*)obj;
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
  baseColor=new SoBaseColor;
  soSep->addChild(baseColor);
  baseColor->rgb.setHSVValue(diffuseColor[0]>0?diffuseColor[0]:0, diffuseColor[1], diffuseColor[2]);

  // GUI
  if(!clone) {
    FloatEditor *minimalColorValueEditor=new FloatEditor(properties, QIcon(), "Minimal color value");
    minimalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMinimalColorValue, &OpenMBV::DynamicColoredBody::setMinimalColorValue);

    FloatEditor *maximalColorValueEditor=new FloatEditor(properties, QIcon(), "Maximal color value");
    maximalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMaximalColorValue, &OpenMBV::DynamicColoredBody::setMaximalColorValue);

    ColorEditor *diffuseColorValue=new ColorEditor(properties, QIcon(), "Diffuse color", true);
    diffuseColorValue->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getDiffuseColor, &OpenMBV::DynamicColoredBody::setDiffuseColor);

    FloatEditor *transparencyValueEditor=new FloatEditor(properties, QIcon(), "Transparency value");
    transparencyValueEditor->setRange(0, 1);
    transparencyValueEditor->setStep(0.1);
    transparencyValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getTransparency, &OpenMBV::DynamicColoredBody::setTransparency);
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
  mat->specularColor.setHSVValue(hue, 0.7*s, v);
}

QString DynamicColoredBody::getInfo() {
  return Body::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Color:</b> %1").arg(getColor());
}

}
