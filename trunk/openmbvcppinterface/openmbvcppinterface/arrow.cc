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
using namespace OpenMBV;

Arrow::Arrow() : DynamicColoredBody(), pathStr("false"),
  data(0),
  headDiameter(0.5),
  headLength(0.75),
  diameter(0.25),
  scaleLength(1),
  type(toHead) {
}

Arrow::~Arrow() {
  if(!hdf5LinkBody && data) { delete data; data=0; }
}

TiXmlElement *Arrow::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=DynamicColoredBody::writeXMLFile(parent);
  addAttribute(e, "path", pathStr, "false");
  addElementText(e, "diameter", diameter);
  addElementText(e, "headDiameter", headDiameter);
  addElementText(e, "headLength", headLength);
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
  addElementText(e, "type", "\""+typeStr+"\"");
  addElementText(e, "scaleLength", scaleLength);
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
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    data->open(*hdf5Group,"data");
  }
}

void Arrow::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(element->Attribute("path") && 
     (element->Attribute("path")==string("true") || element->Attribute("path")==string("1")))
    setPath(true);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"diameter");
  setDiameter(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"headDiameter");
  setHeadDiameter(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"headLength");
  setHeadLength(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"type");
  string typeStr=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
  if(typeStr=="line")            setType(line);
  if(typeStr=="fromHead")        setType(fromHead);
  if(typeStr=="toHead")          setType(toHead);
  if(typeStr=="bothHeads")       setType(bothHeads);
  if(typeStr=="fromDoubleHead")  setType(fromDoubleHead);
  if(typeStr=="toDoubleHead")    setType(toDoubleHead);
  if(typeStr=="bothDoubleHeads") setType(bothDoubleHeads);
  e=element->FirstChildElement(OPENMBVNS"scaleLength");
  setScaleLength(getDouble(e));
}
