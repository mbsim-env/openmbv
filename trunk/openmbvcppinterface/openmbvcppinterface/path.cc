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
#include <openmbvcppinterface/path.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Path::Path() : Body(), data(NULL), color(vector<double>(3,1)) {
}

Path::~Path() {
  if(!hdf5LinkBody && data) { delete data; data=0; }
}

TiXmlElement* Path::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=Body::writeXMLFile(parent);
  addElementText(e, "color", color);
  return 0;
}

void Path::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("x");
    columns.push_back("y");
    columns.push_back("z");
    data->create(*hdf5Group,"data",columns);
  }
}

void Path::openHDF5File() {
  Body::openHDF5File();
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

void Path::initializeUsingXML(TiXmlElement *element) {
  Body::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"color");
  setColor(getVec(e,3));
}
