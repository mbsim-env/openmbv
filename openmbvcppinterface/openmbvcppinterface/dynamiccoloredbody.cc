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

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <fstream>
#include <cmath>

using namespace std;
using namespace OpenMBV;

DynamicColoredBody::DynamicColoredBody() : Body(),
  minimalColorValue(0),
  maximalColorValue(1),
  staticColor(NAN),
  dynamicColor(NAN) {
}

DynamicColoredBody::~DynamicColoredBody() {}

TiXmlElement* DynamicColoredBody::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=Body::writeXMLFile(parent);
  addElementText(e, "minimalColorValue", minimalColorValue, 0);
  addElementText(e, "maximalColorValue", maximalColorValue, 1);
  addElementText(e, "staticColor", staticColor, nan(""));
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
  if(e)
    setStaticColor(getDouble(e));
}
