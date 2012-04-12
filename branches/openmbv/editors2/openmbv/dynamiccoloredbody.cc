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

DynamicColoredBody::DynamicColoredBody(OpenMBV::Object *obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind), color(0), oldColor(nan("")) {
  OpenMBV::DynamicColoredBody *dcb=(OpenMBV::DynamicColoredBody*)obj;
  // read XML
  minimalColorValue=dcb->getMinimalColorValue();
  maximalColorValue=dcb->getMaximalColorValue();
  staticColor=dcb->getStaticColor();

  // GUI
  minimalColorValueEditor=new FloatEditor(this, QIcon(), "Minimal color value");
  minimalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMinimalColorValue, &OpenMBV::DynamicColoredBody::setMinimalColorValue);

  maximalColorValueEditor=new FloatEditor(this, QIcon(), "Maximal color value");
  maximalColorValueEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getMaximalColorValue, &OpenMBV::DynamicColoredBody::setMaximalColorValue);

  staticColorEditor=new FloatEditor(this, QIcon(), "Static color value");
  staticColorEditor->setNaNText("not used");
  staticColorEditor->setOpenMBVParameter(dcb, &OpenMBV::DynamicColoredBody::getStaticColor, &OpenMBV::DynamicColoredBody::setStaticColor);
}

void DynamicColoredBody::setColor(SoMaterial *mat, double col, SoBaseColor *base) {
  if(oldColor!=col) {
    color=col;
    oldColor=col;
    double m=1/(maximalColorValue-minimalColorValue);
    col=m*col-m*minimalColorValue;
    if(col<0) col=0;
    if(col>1) col=1;
    if(base) base->rgb.setHSVValue((1-col)*2/3,1,1);
    if(mat) mat->diffuseColor.setHSVValue((1-col)*2/3,1,1);
    if(mat) mat->specularColor.setHSVValue((1-col)*2/3,0.7,1);
  }
}

QString DynamicColoredBody::getInfo() {
  return Body::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>Color:</b> %1").arg(getColor());
}

QMenu* DynamicColoredBody::createMenu() {
  QMenu* menu=Body::createMenu();
  menu->addSeparator()->setText("Properties from: DynamicColoredBody");
  menu->addAction(minimalColorValueEditor->getAction());
  menu->addAction(maximalColorValueEditor->getAction());
  menu->addAction(staticColorEditor->getAction());
  return menu;
}
