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
#include "openmbvcppinterface/object.h"
#include "openmbvcppinterface/group.h"
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <boost/core/demangle.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

// we use none signaling (quiet) NaN values for double in OpenMBVC++Interface -> Throw compile error if these do not exist.
static_assert(numeric_limits<double>::has_quiet_NaN, "This platform does not support quiet NaN for double.");

Object::Object() : name("NOTSET"), enableStr("true"), boundingBoxStr("false"), ID(""), environmentStr("false") {
}

Object::~Object() = default;

string Object::getFullName() {
  if(fullName.empty()) {
    std::shared_ptr<Group> p=parent.lock();
    if(p)
      fullName = p->getFullName()+"/"+name;
    else
      fullName = name;
  }
  return fullName;
}

void Object::initializeUsingXML(DOMElement *element) {
  setName(E(element)->getAttribute("name"));
  if(E(element)->hasAttribute("enable") && 
     (E(element)->getAttribute("enable")=="false" || E(element)->getAttribute("enable")=="0"))
    setEnable(false);
  if(E(element)->hasAttribute("boundingBox") && 
     (E(element)->getAttribute("boundingBox")=="true" || E(element)->getAttribute("boundingBox")=="1"))
    setBoundingBox(true);
  if(E(element)->hasAttribute("environment") && 
     (E(element)->getAttribute("environment")=="true" || E(element)->getAttribute("environment")=="1"))
    setEnvironment(true);

  DOMProcessingInstruction *ID = E(element)->getFirstProcessingInstructionChildNamed("OPENMBV_ID");
  if(ID)
    setID(X()%ID->getData());
}

DOMElement *Object::writeXMLFile(DOMNode *parent) {
  DOMDocument *doc=parent->getNodeType()==DOMNode::DOCUMENT_NODE ? static_cast<DOMDocument*>(parent) : parent->getOwnerDocument();

  // get type name of this object, without namespace since everything must be in the same namespace
  string typeName(boost::core::demangle(typeid(*this).name()));
  typeName=typeName.substr(typeName.rfind(':')+1);

  DOMElement *e=D(doc)->createElement(OPENMBV%typeName);
  parent->insertBefore(e, nullptr);
  E(e)->setAttribute("name", name);
  E(e)->setAttribute("enable", enableStr);
  E(e)->setAttribute("boundingBox", boundingBoxStr);
  if(environmentStr=="true") E(e)->setAttribute("environment", environmentStr);
  if(!ID.empty()) {
    DOMDocument *doc=parent->getOwnerDocument();
    DOMProcessingInstruction *id=doc->createProcessingInstruction(u"OPENMBV_ID", X()%ID);
    e->insertBefore(id, nullptr);
  }
  return e;
}

std::shared_ptr<Group> Object::getTopLevelGroup() {
  return parent.lock()->getTopLevelGroup();
}

bool Object::getEnvironment() {
  if(environmentStr=="true")
    return true;
  auto p=parent.lock();
  if(p)
    return p->getEnvironment();
  return false;
}

}
