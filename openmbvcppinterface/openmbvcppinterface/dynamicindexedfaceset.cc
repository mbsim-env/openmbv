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
#include <openmbvcppinterface/dynamicindexedfaceset.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(DynamicIndexedFaceSet, OPENMBV%"DynamicIndexedFaceSet")

DOMElement* DynamicIndexedFaceSet::writeXMLFile(DOMNode *parent) {
  DOMElement *e=FlexibleBody::writeXMLFile(parent);
  vector<int> indices1based(indices.size());
  transform(indices.begin(), indices.end(), indices1based.begin(), [](int a){ return a+1; });
  E(e)->addElementText(OPENMBV%"indices", indices1based);
  return e;
}

void DynamicIndexedFaceSet::initializeUsingXML(DOMElement *element) {
  FlexibleBody::initializeUsingXML(element);
  auto e=E(element)->getFirstElementChildNamed(OPENMBV%"indices");
  vector<int> indices1based=E(e)->getText<vector<int>>();
  indices.resize(indices1based.size());
  transform(indices1based.begin(), indices1based.end(), indices.begin(), [](int a){ return a-1; });
}

}
