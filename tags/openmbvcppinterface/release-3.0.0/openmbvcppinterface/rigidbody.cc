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

#include <openmbvcppinterface/rigidbody.h>
#include <iostream>
#include <fstream>
#include <openmbvcppinterface/group.h>
#include <H5Cpp.h>

using namespace std;
using namespace OpenMBV;

RigidBody::RigidBody() : DynamicColoredBody(),
  initialTranslation(3, 0),
  initialRotation(3, 0),
  scaleFactor(1),
  data(0) {
}

RigidBody::~RigidBody() {
  if(!hdf5LinkBody && data) delete data;
}

void RigidBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  DynamicColoredBody::writeXMLFile(xmlFile, indent);
  xmlFile<<indent<<"<initialTranslation>["<<initialTranslation[0]<<";"
                                       <<initialTranslation[1]<<";"
                                       <<initialTranslation[2]<<"]</initialTranslation>"<<endl;
  xmlFile<<indent<<"<initialRotation>["<<initialRotation[0]<<";"
                                    <<initialRotation[1]<<";"
                                    <<initialRotation[2]<<"]</initialRotation>"<<endl;
  xmlFile<<indent<<"<scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
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

void RigidBody::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"initialTranslation");
  setInitialTranslation(getVec(e,3));
  e=element->FirstChildElement(OPENMBVNS"initialRotation");
  setInitialRotation(getVec(e,3));
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  setScaleFactor(getDouble(e));
}
