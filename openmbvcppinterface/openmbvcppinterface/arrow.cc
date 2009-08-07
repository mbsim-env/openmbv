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

#include <openmbvcppinterface/arrow.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Arrow::Arrow() : DynamicColoredBody(),
  diameter(1),
  headDiameter(0.5),
  headLength(0.75),
  type(toHead),
  scaleLength(1) {
}

void Arrow::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Arrow name=\""<<name<<"\">"<<endl;
    DynamicColoredBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <diameter>"<<diameter<<"</diameter>"<<endl;
    xmlFile<<indent<<"  <headDiameter>"<<headDiameter<<"</headDiameter>"<<endl;
    xmlFile<<indent<<"  <headLength>"<<headLength<<"</headLength>"<<endl;
    string typeStr;
    switch(type) {
      case line: typeStr="line"; break;
      case fromHead: typeStr="fromHead"; break;
      case toHead: typeStr="toHead"; break;
      case bothHeads: typeStr="bothHeads"; break;
    }
    xmlFile<<indent<<"  <type>\""<<typeStr<<"\"</type>"<<endl;
    xmlFile<<indent<<"  <scaleLength>"<<scaleLength<<"</scaleLength>"<<endl;
  xmlFile<<indent<<"</Arrow>"<<endl;
}

void Arrow::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("toPoint x");
    columns.push_back("toPoint x");
    columns.push_back("toPoint x");
    columns.push_back("delte x");
    columns.push_back("delte y");
    columns.push_back("delte z");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}

void Arrow::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"diameter");
  setDiameter(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"headDiameter");
  setHeadDiameter(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"headLength");
  setHeadLength(toVector(e->GetText())[0]);
  e=element->FirstChildElement(OPENMBVNS"type");
  string typeStr=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
  if(typeStr=="line") setType(line);
  if(typeStr=="fromHead") setType(fromHead);
  if(typeStr=="toHead") setType(toHead);
  if(typeStr=="bothHeads") setType(bothHeads);
  e=element->FirstChildElement(OPENMBVNS"scaleLength");
  setScaleLength(toVector(e->GetText())[0]);
}
