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
#include <openmbvcppinterface/rigidspineextrusion.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(RigidSpineExtrusion, OPENMBV%"RigidSpineExtrusion")

RigidSpineExtrusion::RigidSpineExtrusion() = default;

RigidSpineExtrusion::~RigidSpineExtrusion() = default;

DOMElement* RigidSpineExtrusion::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);

  if(contour) PolygonPoint::serializePolygonPointContour(e, contour);

  E(e)->addElementText(OPENMBV%"counterClockWise", ccw);

  vector<vector<double>> m;
  m.reserve(spine.size());
  for(const auto &s : spine)
    m.emplace_back(vector<double>{s.x,s.y,s.z,s.alpha,s.beta,s.gamma});
  E(e)->addElementText(OPENMBV%"spine", m);

  return nullptr;
}

void RigidSpineExtrusion::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);

  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"contour");
  setContour(PolygonPoint::initializeUsingXML(e));

  setCounterClockWise(true);
  e=E(element)->getFirstElementChildNamed(OPENMBV%"counterClockWise");
  if(e) setCounterClockWise(E(e)->getText<bool>());

  e=E(element)->getFirstElementChildNamed(OPENMBV%"spine");
  auto m = E(e)->getText<vector<vector<double>>>();
  spine.clear();
  for(const auto& r : m)
    spine.emplace_back(r[0],r[1],r[2],r[3],r[4],r[5]);
}

}
