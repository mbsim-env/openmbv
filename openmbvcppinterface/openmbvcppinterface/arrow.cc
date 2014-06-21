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
#include <openmbvcppinterface/arrow.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Arrow, OPENMBV%"Arrow")

Arrow::Arrow() : DynamicColoredBody(), pathStr("false"),
  data(0),
  headDiameter(0.5),
  headLength(0.75),
  diameter(0.25),
  scaleLength(1),
  type(toHead),
  referencePoint(toPoint) {
}

Arrow::~Arrow() {
  if(!hdf5LinkBody && data) { delete data; data=0; }
}

DOMElement *Arrow::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  addAttribute(e, "path", pathStr, "false");
  addElementText(e, OPENMBV%"diameter", diameter);
  addElementText(e, OPENMBV%"headDiameter", headDiameter);
  addElementText(e, OPENMBV%"headLength", headLength);
  string typeStr;
  switch(type) {
    case line:            typeStr="line";            break;
    case fromHead:        typeStr="fromHead";        break;
    case toHead:          typeStr="toHead";          break;
    case bothHeads:       typeStr="bothHeads";       break;
    case fromDoubleHead:  typeStr="fromDoubleHead";  break;
    case toDoubleHead:    typeStr="toDoubleHead";    break;
    case bothDoubleHeads: typeStr="bothDoubleHeads"; break;
  }
  addElementText(e, OPENMBV%"type", "'"+typeStr+"'");
  string referencePointStr;
  switch(referencePoint) {
    case toPoint:   referencePointStr="toPoint";   break;
    case fromPoint: referencePointStr="fromPoint"; break;
    case midPoint:  referencePointStr="midPoint";  break;
  }
  addElementText(e, OPENMBV%"referencePoint", "'"+referencePointStr+"'");
  addElementText(e, OPENMBV%"scaleLength", scaleLength);
  return 0;
}

void Arrow::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("toPoint x");
    columns.push_back("toPoint y");
    columns.push_back("toPoint z");
    columns.push_back("delta x");
    columns.push_back("delta y");
    columns.push_back("delta z");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}

void Arrow::openHDF5File() {
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

void Arrow::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(E(element)->hasAttribute("path") && 
     (E(element)->getAttribute("path")=="true" || E(element)->getAttribute("path")=="1"))
    setPath(true);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"diameter");
  setDiameter(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"headDiameter");
  setHeadDiameter(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"headLength");
  setHeadLength(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"type");
  string name = X()%E(e)->getFirstTextChild()->getData();
  string typeStr=name.substr(1,name.length()-2);
  if(typeStr=="line")            setType(line);
  if(typeStr=="fromHead")        setType(fromHead);
  if(typeStr=="toHead")          setType(toHead);
  if(typeStr=="bothHeads")       setType(bothHeads);
  if(typeStr=="fromDoubleHead")  setType(fromDoubleHead);
  if(typeStr=="toDoubleHead")    setType(toDoubleHead);
  if(typeStr=="bothDoubleHeads") setType(bothDoubleHeads);
  e=E(element)->getFirstElementChildNamed(OPENMBV%"referencePoint");
  if(e) {
    string name = X()%E(e)->getFirstTextChild()->getData();
    string referencePointStr=name.substr(1,name.length()-2);
    if(referencePointStr=="toPoint")   setReferencePoint(toPoint);
    if(referencePointStr=="fromPoint") setReferencePoint(fromPoint);
    if(referencePointStr=="midPoint")  setReferencePoint(midPoint);
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleLength");
  setScaleLength(getDouble(e));
}

}
