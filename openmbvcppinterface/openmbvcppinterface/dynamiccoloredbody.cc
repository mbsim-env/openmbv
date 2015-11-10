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
#include "openmbvcppinterface/dynamiccoloredbody.h"
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

DynamicColoredBody::DynamicColoredBody() : Body(),
  minimalColorValue(0),
  maximalColorValue(1),
  dynamicColor(numeric_limits<double>::quiet_NaN()),
  diffuseColor(vector<double>(3)),
  transparency(0) {
  vector<double> hsv(3);
  hsv[0]=-1;
  hsv[1]=1;
  hsv[2]=1;
  diffuseColor=hsv;
}

DynamicColoredBody::~DynamicColoredBody() {}

DOMElement* DynamicColoredBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  addElementText(e, OPENMBV%"minimalColorValue", minimalColorValue, 0);
  addElementText(e, OPENMBV%"maximalColorValue", maximalColorValue, 1);
  if(diffuseColor[0]>=0 || diffuseColor[1]!=1 || diffuseColor[2]!=1)
    addElementText(e, OPENMBV%"diffuseColor", diffuseColor);
  addElementText(e, OPENMBV%"transparency", transparency, 0);
  return e;
}

void DynamicColoredBody::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"minimalColorValue");
  if(e)
    setMinimalColorValue(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"maximalColorValue");
  if(e)
    setMaximalColorValue(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"diffuseColor");
  if(e)
    setDiffuseColor(getVec(e, 3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"transparency");
  if(e)
    setTransparency(getDouble(e));
}

}
