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
#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <fstream>
#include <cmath>

using namespace std;
using namespace MBXMLUtils;
using namespace OpenMBV;

DynamicColoredBody::DynamicColoredBody() : Body(),
  minimalColorValue(0),
  maximalColorValue(1),
  staticColor(NAN),
  dynamicColor(NAN),
  diffuseColor(vector<double>(3)),
  transparency(0) {
  vector<double> hsv(3);
  hsv[0]=-1;
  hsv[1]=1;
  hsv[2]=1;
  set(diffuseColor,hsv);
}

DynamicColoredBody::~DynamicColoredBody() {}

TiXmlElement* DynamicColoredBody::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=Body::writeXMLFile(parent);
  addElementText(e, OPENMBVNS"minimalColorValue", minimalColorValue, 0);
  addElementText(e, OPENMBVNS"maximalColorValue", maximalColorValue, 1);
  addElementText(e, OPENMBVNS"staticColor", staticColor, nan(""));
  if(diffuseColor.getValue()[0]>=0 || diffuseColor.getValue()[1]!=1 || diffuseColor.getValue()[2]!=1)
    addElementText(e, OPENMBVNS"diffuseColor", diffuseColor);
  addElementText(e, OPENMBVNS"transparency", transparency, 0);
  return e;
}

void DynamicColoredBody::initializeUsingXML(TiXmlElement *element) {
  Body::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"minimalColorValue");
  if(e)
    setMinimalColorValue(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"maximalColorValue");
  if(e)
    setMaximalColorValue(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"staticColor");
  if(e) {
    MBXMLUtils::Deprecated::registerMessage("<ombv:staticColor> is deprecated, use <ombv:diffuseColor> instead.", e);
    set(staticColor, getDouble(e));// do not call setStaticColor here to avoid another deprecated message
  }
  e=element->FirstChildElement(OPENMBVNS"diffuseColor");
  if(e)
    setDiffuseColor(getVec(e, 3));
  e=element->FirstChildElement(OPENMBVNS"transparency");
  if(e)
    setTransparency(getDouble(e));
}
