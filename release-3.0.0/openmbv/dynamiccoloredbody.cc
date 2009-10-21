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

using namespace std;

DynamicColoredBody::DynamicColoredBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : Body(element, h5Parent, parentItem, soParent), oldColor(nan("")), color(0) {
  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"minimalColorValue");
  if(e)
    minimalColorValue=toVector(e->GetText())[0];
  else
    minimalColorValue=0;
  e=element->FirstChildElement(OPENMBVNS"maximalColorValue");
  if(e)
    maximalColorValue=toVector(e->GetText())[0];
  else
    maximalColorValue=1;
  e=element->FirstChildElement(OPENMBVNS"staticColor");
  if(e) {
    QByteArray tmp(e->GetText()); 
    staticColor=tmp.toDouble();
  }
  else
    staticColor=nan("");
}

void DynamicColoredBody::setColor(SoMaterial *mat, double col) {
  if(oldColor!=col) {
    color=col;
    oldColor=col;
    double m=1/(maximalColorValue-minimalColorValue);
    col=m*col-m*minimalColorValue;
    mat->diffuseColor.setHSVValue((1-col)*2/3,1,1);
    mat->specularColor.setHSVValue((1-col)*2/3,0.7,1);
  }
}

QString DynamicColoredBody::getInfo() {
  return Body::getInfo()+
         QString("-----<br/>")+
         QString("<b>Color:</b> %1<br/>").arg(getColor());
}

