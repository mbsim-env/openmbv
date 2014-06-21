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
#include <H5Cpp.h>
#include <openmbvcppinterface/compoundrigidbody.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

RigidBody::RigidBody() : DynamicColoredBody(), localFrameStr("false"), referenceFrameStr("false"), pathStr("false"), draggerStr("false"), 
  initialTranslation(vector<double>(3, 0)),
  initialRotation(vector<double>(3, 0)),
  scaleFactor(1),
  data(0),
  compound(0) {
}

RigidBody::~RigidBody() {
  if(!hdf5LinkBody && data) delete data;
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
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("x");
    columns.push_back("y");
    columns.push_back("z");
    columns.push_back("alpha");
    columns.push_back("beta");
    columns.push_back("gamma");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}

void RigidBody::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    try {
      data->open(*hdf5Group,"data");
    }
    catch(...) {
      delete data;
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

Group* RigidBody::getSeparateGroup() {
  return compound?compound->parent->getSeparateGroup():parent->getSeparateGroup();
}

Group* RigidBody::getTopLevelGroup() {
  return compound?compound->parent->getTopLevelGroup():parent->getTopLevelGroup();
}

string RigidBody::getFullName(bool includingFileName, bool stopAtSeparateFile) {
  if(compound)
    return compound->getFullName(includingFileName, stopAtSeparateFile)+"/"+name;
  else
    return DynamicColoredBody::getFullName(includingFileName, stopAtSeparateFile);
}

void RigidBody::destroy() const {
  // a RigidBody may be part of a compoundrigidbody. If so delete from this if not treat it like other objects

  // delete this rigidBody from compound if compound exists
  if(compound) {
    for(vector<RigidBody*>::iterator i=compound->rigidBody.begin(); i!=compound->rigidBody.end(); i++)
      if(*i==this) {
        compound->rigidBody.erase(i);
        break;
      }
  }
  else {
    DynamicColoredBody::destroy();
    return;
  }
  // destroy this rigidBody
  delete this;
}

}
