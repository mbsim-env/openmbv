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
#include "openmbvcppinterface/dynamiccoloredbody.h"
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

DynamicColoredBody::DynamicColoredBody() : 
  
  dynamicColor(numeric_limits<double>::quiet_NaN()),
  diffuseColor(vector<double>(3))
  {
  vector<double> hsv(3);
  hsv[0]=-1;
  hsv[1]=1;
  hsv[2]=1;
  diffuseColor=hsv;
}

DynamicColoredBody::~DynamicColoredBody() = default;

DOMElement* DynamicColoredBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"minimalColorValue", minimalColorValue);
  E(e)->addElementText(OPENMBV%"maximalColorValue", maximalColorValue);
  if(diffuseColor[0]>=0 || diffuseColor[1]!=1 || diffuseColor[2]!=1)
    E(e)->addElementText(OPENMBV%"diffuseColor", diffuseColor);
  E(e)->addElementText(OPENMBV%"transparency", transparency);
  return e;
}

void DynamicColoredBody::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"minimalColorValue");
  if(e)
    setMinimalColorValue(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"maximalColorValue");
  if(e)
    setMaximalColorValue(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"diffuseColor");
  if(e)
    setDiffuseColor(E(e)->getText<vector<double>>(3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"transparency");
  if(e)
    setTransparency(E(e)->getText<double>());
}

}
