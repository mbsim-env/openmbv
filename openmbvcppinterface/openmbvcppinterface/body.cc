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
#include <openmbvcppinterface/body.h>
#include <iostream>
#include <fstream>
#include <openmbvcppinterface/group.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

Body::Body() :  outLineStr("true"), shilouetteEdgeStr("false") {
}

DOMElement* Body::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Object::writeXMLFile(parent);
  E(e)->setAttribute("outLine", outLineStr);
  E(e)->setAttribute("shilouetteEdge", shilouetteEdgeStr);
  string dm;
  switch(drawMethod) {
    case filled: dm="filled"; break;
    case lines: dm="lines"; break;
    case points: dm="points"; break;
  }
  E(e)->setAttribute("drawMethod", dm);
  E(e)->setAttribute("pointSize", pointSize);
  E(e)->setAttribute("lineWidth", lineWidth);
  return e;
}

void Body::createHDF5File() {
  std::shared_ptr<Group> p=parent.lock();
  hdf5Group=p->hdf5Group->createChildObject<H5::Group>(name)();
}

void Body::openHDF5File() {
  hdf5Group=nullptr;
  try {
    std::shared_ptr<Group> p=parent.lock();
    hdf5Group=p->hdf5Group->openChildObject<H5::Group>(name);
  }
  catch(...) {
    msg(Warn)<<"Unable to open the HDF5 Group '"<<name<<"'"<<endl;
  }
}

std::string Body::getRelPathTo(const std::shared_ptr<Body> &destBody) {
  // create relative path to destination
  string dest=destBody->getFullName();
  string src=getFullName();
  string reldest;
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

void Body::initializeUsingXML(DOMElement *element) {
  Object::initializeUsingXML(element);
  if(E(element)->hasAttribute("outLine") && 
     (E(element)->getAttribute("outLine")=="false" || E(element)->getAttribute("outLine")=="0"))
    setOutLine(false);
  if(E(element)->hasAttribute("shilouetteEdge") && 
     (E(element)->getAttribute("shilouetteEdge")=="true" || E(element)->getAttribute("shilouetteEdge")=="1"))
    setShilouetteEdge(true);
  if(E(element)->hasAttribute("drawMethod")) {
    if(E(element)->getAttribute("drawMethod")=="filled") setDrawMethod(filled);
    if(E(element)->getAttribute("drawMethod")=="lines") setDrawMethod(lines);
    if(E(element)->getAttribute("drawMethod")=="points") setDrawMethod(points);
  }
  if(E(element)->hasAttribute("pointSize"))
   setPointSize(stod(E(element)->getAttribute("pointSize")));
  if(E(element)->hasAttribute("lineWidth"))
   setLineWidth(stod(E(element)->getAttribute("lineWidth")));
}

}
