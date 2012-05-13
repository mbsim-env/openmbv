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
#include <openmbvcppinterface/coilspring.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

CoilSpring::CoilSpring() : DynamicColoredBody(),
  data(0),
  springRadius(1),
  crossSectionRadius(-1),
  scaleFactor(1),
  numberOfCoils(3),
  nominalLength(-1),
  type(tube) {
}

CoilSpring::~CoilSpring() {
  if(!hdf5LinkBody && data) { delete data; data=0; }
}

TiXmlElement *CoilSpring::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=DynamicColoredBody::writeXMLFile(parent);
  string typeStr;
  switch(type) {
    case tube: typeStr="tube"; break;
    case scaledTube: typeStr="scaledTube"; break;
    case polyline: typeStr="polyline"; break;
  }
  addElementText(e, "type", "\""+typeStr+"\"");
  addElementText(e, "numberOfCoils", numberOfCoils);
  addElementText(e, "springRadius", springRadius);
  addElementText(e, "crossSectionRadius", crossSectionRadius);
  addElementText(e, "nominalLength", nominalLength);
  addElementText(e, "scaleFactor", scaleFactor);
  return 0;
}

void CoilSpring::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("fromPoint x");
    columns.push_back("fromPoint y");
    columns.push_back("fromPoint z");
    columns.push_back("toPoint x");
    columns.push_back("toPoint y");
    columns.push_back("toPoint z");
    columns.push_back("color");
    data->create(*hdf5Group,"data",columns);
  }
}

void CoilSpring::openHDF5File() {
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
      cout<<"WARNING: Unable to open the HDF5 Dataset 'data'"<<endl;
    }
  }
}

void CoilSpring::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"type");
  if(e) {
    string typeStr=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
    if(typeStr=="tube") setType(tube);
    if(typeStr=="scaledTube") setType(scaledTube);
    if(typeStr=="polyline") setType(polyline);
  }
  e=element->FirstChildElement(OPENMBVNS"numberOfCoils");
  setNumberOfCoils(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"springRadius");
  setSpringRadius(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"crossSectionRadius");
  setCrossSectionRadius(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"nominalLength");
  if(e) setNominalLength(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  setScaleFactor(getDouble(e));
}
