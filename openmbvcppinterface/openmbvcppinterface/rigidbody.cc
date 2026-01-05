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
#include <openmbvcppinterface/rigidbody.h>
#include <iostream>
#include <fstream>
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/compoundrigidbody.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

RigidBody::RigidBody() :  localFrameStr("false"), referenceFrameStr("false"), pathStr("false"), draggerStr("false"), 
  initialTranslation(vector<double>(3, 0)),
  initialRotation(vector<double>(3, 0))
  {
}

RigidBody::~RigidBody() = default;

DOMElement* RigidBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  E(e)->setAttribute("localFrame", localFrameStr);
  E(e)->setAttribute("referenceFrame", referenceFrameStr);
  E(e)->setAttribute("path", pathStr);
  E(e)->setAttribute("dragger", draggerStr);
  E(e)->addElementText(OPENMBV%"initialTranslation", initialTranslation);
  E(e)->addElementText(OPENMBV%"initialRotation", initialRotation);
  E(e)->addElementText(OPENMBV%"scaleFactor", scaleFactor);
  return e;
}

void RigidBody::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<Float> >("data")(8);
  vector<string> columns;
  columns.emplace_back("Time");
  columns.emplace_back("x");
  columns.emplace_back("y");
  columns.emplace_back("z");
  columns.emplace_back("alpha");
  columns.emplace_back("beta");
  columns.emplace_back("gamma");
  columns.emplace_back("color");
  data->setColumnLabel(columns);
}

void RigidBody::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<Float> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
}

void RigidBody::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(E(element)->hasAttribute("localFrame") && 
     (E(element)->getAttribute("localFrame")=="true" || E(element)->getAttribute("localFrame")=="1"))
    setLocalFrame(true);
  if(E(element)->hasAttribute("referenceFrame") && 
     (E(element)->getAttribute("referenceFrame")=="true" || E(element)->getAttribute("referenceFrame")=="1"))
    setReferenceFrame(true);
  if(E(element)->hasAttribute("path") && 
     (E(element)->getAttribute("path")=="true" || E(element)->getAttribute("path")=="1"))
    setPath(true);
  if(E(element)->hasAttribute("dragger") && 
     (E(element)->getAttribute("dragger")=="true" || E(element)->getAttribute("dragger")=="1"))
    setDragger(true);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"initialTranslation");
  if(e) setInitialTranslation(E(e)->getText<vector<double>>(3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"initialRotation");
  if(e) setInitialRotation(E(e)->getText<vector<double>>(3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  if(e) setScaleFactor(E(e)->getText<double>());
}

std::shared_ptr<Group> RigidBody::getTopLevelGroup() {
  std::shared_ptr<CompoundRigidBody> c=compound.lock();
  return c?c->parent.lock()->getTopLevelGroup():parent.lock()->getTopLevelGroup();
}

string RigidBody::getFullName() {
  if(fullName.empty()) {
    std::shared_ptr<CompoundRigidBody> c=compound.lock();
    if(c)
      fullName = c->getFullName()+"/"+name;
    else
      fullName = DynamicColoredBody::getFullName();
  }
  return fullName;
}

}
