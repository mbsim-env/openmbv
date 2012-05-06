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
#include <openmbvcppinterface/simpleparameter.h>
#include <openmbvcppinterface/compoundrigidbody.h>

using namespace std;
using namespace OpenMBV;

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

TiXmlElement* RigidBody::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=DynamicColoredBody::writeXMLFile(parent);
  addAttribute(e, "localFrame", localFrameStr, "false");
  addAttribute(e, "referenceFrame", referenceFrameStr, "false");
  addAttribute(e, "path", pathStr, "false");
  addAttribute(e, "dragger", draggerStr, "false");
  addElementText(e, "initialTranslation", initialTranslation);
  addElementText(e, "initialRotation", initialRotation);
  addElementText(e, "scaleFactor", scaleFactor);
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
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    data->open(*hdf5Group,"data");
  }
}

void RigidBody::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(element->Attribute("localFrame") && 
     (element->Attribute("localFrame")==string("true") || element->Attribute("localFrame")==string("1")))
    setLocalFrame(true);
  if(element->Attribute("referenceFrame") && 
     (element->Attribute("referenceFrame")==string("true") || element->Attribute("referenceFrame")==string("1")))
    setReferenceFrame(true);
  if(element->Attribute("path") && 
     (element->Attribute("path")==string("true") || element->Attribute("path")==string("1")))
    setPath(true);
  if(element->Attribute("dragger") && 
     (element->Attribute("dragger")==string("true") || element->Attribute("dragger")==string("1")))
    setDragger(true);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"initialTranslation");
  setInitialTranslation(getVec(e,3));
  e=element->FirstChildElement(OPENMBVNS"initialRotation");
  setInitialRotation(getVec(e,3));
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  setScaleFactor(getDouble(e));
}

Group* RigidBody::getSeparateGroup() {
  return compound?compound->parent->getSeparateGroup():parent->getSeparateGroup();
}

Group* RigidBody::getTopLevelGroup() {
  return compound?compound->parent->getTopLevelGroup():parent->getTopLevelGroup();
}

string RigidBody::getFullName() {
  if(compound)
    return compound->getFullName()+"/"+name;
  else
    return DynamicColoredBody::getFullName();
}
