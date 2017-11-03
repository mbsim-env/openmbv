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
#include "openmbvcppinterface/object.h"
#include "openmbvcppinterface/group.h"
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <assert.h>
#include <cmath>
#include <limits>

#if BOOST_VERSION >= 105600
  #include <boost/core/demangle.hpp>
#else
  #include <cxxabi.h>
  #ifndef BOOST_CORE_DEMANGLE_REPLACEMENT
  #define BOOST_CORE_DEMANGLE_REPLACEMENT
  namespace boost {
    namespace core {
      inline std::string demangle(const std::string &name) {
        int status;
        char* retc=abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
        if(status!=0) throw std::runtime_error("Cannot demangle c++ symbol.");
        std::string ret(retc);
        free(retc);
        return ret;
      }
    }
  }
  #endif
#endif

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

// we use none signaling (quiet) NaN values for double in OpenMBVC++Interface -> Throw compile error if these do not exist.
static_assert(numeric_limits<double>::has_quiet_NaN, "This platform does not support quiet NaN for double.");

Object::Object() : name("NOTSET"), enableStr("true"), boundingBoxStr("false"), ID(""), selected(false), hdf5Group(0) {
}

Object::~Object() {
}

string Object::getFullName(bool includingFileName, bool stopAtSeparateFile) {
  std::shared_ptr<Group> p=parent.lock();
  if(p)
    return p->getFullName(includingFileName, stopAtSeparateFile)+"/"+name;
  else
    return name;
}

void Object::initializeUsingXML(DOMElement *element) {
  setName(E(element)->getAttribute("name"));
  if(E(element)->hasAttribute("enable") && 
     (E(element)->getAttribute("enable")=="false" || E(element)->getAttribute("enable")=="0"))
    setEnable(false);
  if(E(element)->hasAttribute("boundingBox") && 
     (E(element)->getAttribute("boundingBox")=="false" || E(element)->getAttribute("boundingBox")=="0"))
    setBoundingBox(false);

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
  parent->insertBefore(e, NULL);
  E(e)->setAttribute("name", name);
  E(e)->setAttribute("enable", enableStr);
  E(e)->setAttribute("boundingBox", boundingBoxStr);
  if(!ID.empty()) {
    DOMDocument *doc=parent->getOwnerDocument();
    DOMProcessingInstruction *id=doc->createProcessingInstruction(X()%"OPENMBV_ID", X()%ID);
    e->insertBefore(id, NULL);
  }
  return e;
}

std::shared_ptr<Group> Object::getSeparateGroup() {
  return parent.lock()->getSeparateGroup();
}

std::shared_ptr<Group> Object::getTopLevelGroup() {
  return parent.lock()->getTopLevelGroup();
}

}
