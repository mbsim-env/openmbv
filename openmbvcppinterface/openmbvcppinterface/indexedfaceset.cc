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
#include <openmbvcppinterface/indexedfaceset.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(IndexedFaceSet, OPENMBV%"IndexedFaceSet")

IndexedFaceSet::IndexedFaceSet()  {
}

DOMElement* IndexedFaceSet::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"vertexPositions", vp);

  vector<int> indices1based(indices.size());
  transform(indices.begin(), indices.end(), indices1based.begin(), [](int a){ return a+1; });
  E(e)->addElementText(OPENMBV%"indices", indices1based);

  return nullptr;
}

void IndexedFaceSet::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e=E(element)->getFirstElementChildNamed(OPENMBV%"vertexPositions");
  setVertexPositions(E(e)->getText<vector<vector<double>>>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"indices");
  vector<int> indices1based=E(e)->getText<vector<int>>();
  indices.resize(indices1based.size());
  transform(indices1based.begin(), indices1based.end(), indices.begin(), [](int a){ return a-1; });
}

}
