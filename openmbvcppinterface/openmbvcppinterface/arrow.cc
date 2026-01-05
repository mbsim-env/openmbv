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
#include <openmbvcppinterface/arrow.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Arrow, OPENMBV%"Arrow")

Arrow::Arrow() :  pathStr("false")
  {
}

Arrow::~Arrow() = default;

DOMElement *Arrow::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  E(e)->setAttribute("path", pathStr);
  E(e)->addElementText(OPENMBV%"diameter", diameter);
  E(e)->addElementText(OPENMBV%"headDiameter", headDiameter);
  E(e)->addElementText(OPENMBV%"headLength", headLength);
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
  E(e)->addElementText(OPENMBV%"type", "'"+typeStr+"'");
  string referencePointStr;
  switch(referencePoint) {
    case toPoint:   referencePointStr="toPoint";   break;
    case fromPoint: referencePointStr="fromPoint"; break;
    case midPoint:  referencePointStr="midPoint";  break;
  }
  E(e)->addElementText(OPENMBV%"referencePoint", "'"+referencePointStr+"'");
  E(e)->addElementText(OPENMBV%"scaleLength", scaleLength);
  return nullptr;
}

void Arrow::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<Float> >("data")(8);
  vector<string> columns;
  columns.emplace_back("Time");
  columns.emplace_back("toPoint x");
  columns.emplace_back("toPoint y");
  columns.emplace_back("toPoint z");
  columns.emplace_back("delta x");
  columns.emplace_back("delta y");
  columns.emplace_back("delta z");
  columns.emplace_back("color");
  data->setColumnLabel(columns);
}

void Arrow::openHDF5File() {
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

void Arrow::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(E(element)->hasAttribute("path") && 
     (E(element)->getAttribute("path")=="true" || E(element)->getAttribute("path")=="1"))
    setPath(true);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"diameter");
  setDiameter(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"headDiameter");
  setHeadDiameter(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"headLength");
  setHeadLength(E(e)->getText<double>());
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
  setScaleLength(E(e)->getText<double>());
}

}
