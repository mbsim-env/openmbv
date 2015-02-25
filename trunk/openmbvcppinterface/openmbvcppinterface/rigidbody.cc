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
#include <openmbvcppinterface/rigidbody.h>
#include <iostream>
#include <fstream>
#include <openmbvcppinterface/group.h>
#include <openmbvcppinterface/compoundrigidbody.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

RigidBody::RigidBody() : DynamicColoredBody(), localFrameStr("false"), referenceFrameStr("false"), pathStr("false"), draggerStr("false"), 
  initialTranslation(vector<double>(3, 0)),
  initialRotation(vector<double>(3, 0)),
  scaleFactor(1),
  data(0) {
}

RigidBody::~RigidBody() {
}

DOMElement* RigidBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  addAttribute(e, "localFrame", localFrameStr, "false");
  addAttribute(e, "referenceFrame", referenceFrameStr, "false");
  addAttribute(e, "path", pathStr, "false");
  addAttribute(e, "dragger", draggerStr, "false");
  addElementText(e, OPENMBV%"initialTranslation", initialTranslation);
  addElementText(e, OPENMBV%"initialRotation", initialRotation);
  addElementText(e, OPENMBV%"scaleFactor", scaleFactor);
  return e;
}

void RigidBody::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(8);
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("x");
    columns.push_back("y");
    columns.push_back("z");
    columns.push_back("alpha");
    columns.push_back("beta");
    columns.push_back("gamma");
    columns.push_back("color");
    data->setColumnLabel(columns);
  }
}

void RigidBody::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  if(!hdf5LinkBody) {
    try {
      data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
    }
    catch(...) {
      data=NULL;
      msg(Warn)<<"Unable to open the HDF5 Dataset 'data'"<<endl;
    }
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
  setInitialTranslation(getVec(e,3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"initialRotation");
  setInitialRotation(getVec(e,3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  setScaleFactor(getDouble(e));
}

boost::shared_ptr<Group> RigidBody::getSeparateGroup() {
  boost::shared_ptr<CompoundRigidBody> c=compound.lock();
  return c?c->parent.lock()->getSeparateGroup():parent.lock()->getSeparateGroup();
}

boost::shared_ptr<Group> RigidBody::getTopLevelGroup() {
  boost::shared_ptr<CompoundRigidBody> c=compound.lock();
  return c?c->parent.lock()->getTopLevelGroup():parent.lock()->getTopLevelGroup();
}

string RigidBody::getFullName(bool includingFileName, bool stopAtSeparateFile) {
  boost::shared_ptr<CompoundRigidBody> c=compound.lock();
  if(c)
    return c->getFullName(includingFileName, stopAtSeparateFile)+"/"+name;
  else
    return DynamicColoredBody::getFullName(includingFileName, stopAtSeparateFile);
}

}
