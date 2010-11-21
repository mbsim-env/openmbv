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

#include <openmbvcppinterface/body.h>
#include <iostream>
#include <fstream>
#include <H5Cpp.h>
#include <openmbvcppinterface/group.h>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinynamespace.h"

using namespace std;
using namespace OpenMBV;

Body::Body() : Object(), outLineStr("true"), shilouetteEdgeStr("false"), drawMethod(filled),
  hdf5LinkBody(0), hdf5LinkStr("") {
}

TiXmlElement* Body::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=Object::writeXMLFile(parent);
  addAttribute(e, "outLine", outLineStr, "true");
  addAttribute(e, "shilouetteEdge", shilouetteEdgeStr, "false");
  string dm;
  switch(drawMethod) {
    case filled: dm="filled"; break;
    case lines: dm="lines"; break;
    case points: dm="points"; break;
  }
  addAttribute(e, "drawMethod", dm, "filled");
  if(hdf5LinkBody) {
    TiXmlElement *ee=new TiXmlElement("hdf5Link");
    e->LinkEndChild(ee);
    ee->SetAttribute("ref", getRelPathTo(hdf5LinkBody));
  }
  else if(hdf5LinkStr!="") {
    TiXmlElement *ee=new TiXmlElement("hdf5Link");
    e->LinkEndChild(ee);
    ee->SetAttribute("ref", hdf5LinkStr);
  }
  return e;
}

void Body::createHDF5File() {
  if(!hdf5LinkBody)
    hdf5Group=new H5::Group(parent->hdf5Group->createGroup(name));
  else
    parent->hdf5Group->link(H5L_TYPE_SOFT, getRelPathTo(hdf5LinkBody), name);
}

void Body::openHDF5File() {
  hdf5Group=new H5::Group(parent->hdf5Group->openGroup(name));
}

std::string Body::getRelPathTo(Body* destBody) {
  // create relative path to destination
  string dest=destBody->getFullName();
  string src=getFullName();
  string reldest="";
  while(dest.substr(0,dest.find('/',1))==src.substr(0,src.find('/',1)))  {
    dest=dest.substr(dest.find('/',1));
    src=src.substr(src.find('/',1));
  }
  while((signed)src.find('/',1)>=0) {
    reldest=reldest+"../";
    src=src.substr(src.find('/',1));
  }
  reldest=reldest+dest.substr(1);
  return reldest;
}

void Body::terminate() {
}

void Body::initializeUsingXML(TiXmlElement *element) {
  Object::initializeUsingXML(element);
  TiXmlElement *e;
  if(element->Attribute("outLine") && 
     (element->Attribute("outLine")==string("false") || element->Attribute("outLine")==string("0")))
    setOutLine(false);
  if(element->Attribute("shilouetteEdge") && 
     (element->Attribute("shilouetteEdge")==string("true") || element->Attribute("shilouetteEdge")==string("1")))
    setShilouetteEdge(true);
  if(element->Attribute("drawMethod")) {
    if(element->Attribute("drawMethod")==string("filled")) setDrawMethod(filled);
    if(element->Attribute("drawMethod")==string("lines")) setDrawMethod(lines);
    if(element->Attribute("drawMethod")==string("points")) setDrawMethod(points);
  }
  if((e=element->FirstChildElement(OPENMBVNS"hdf5Link")))
    hdf5LinkStr=e->Attribute("ref");
}
