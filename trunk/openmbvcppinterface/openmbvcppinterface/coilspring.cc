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

#include <openmbvcppinterface/coilspring.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

CoilSpring::CoilSpring() : Body(),
  springRadius(1),
  crossSectionRadius(0.1),
  scaleFactor(1) {
}

void CoilSpring::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<CoilSpring name=\""<<name<<"\">"<<endl;
    Body::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <numberOfCoils>"<<numberOfCoils<<"</numberOfCoils>"<<endl;
    xmlFile<<indent<<"  <springRadius>"<<springRadius<<"</springRadius>"<<endl;
    xmlFile<<indent<<"  <crossSectionRadius>"<<crossSectionRadius<<"</crossSectionRadius>"<<endl;
    xmlFile<<indent<<"  <scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
  xmlFile<<indent<<"</CoilSpring>"<<endl;
}

void CoilSpring::createHDF5File() {
  Body::createHDF5File();
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

void CoilSpring::initializeUsingXML(TiXmlElement *element) {
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"numberOfCoils");
  setNumberOfCoils(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"springRadius");
  setSpringRadius(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"crossSectionRadius");
  setCrossSectionRadius(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  setScaleFactor(atof(e->GetText()));
}
