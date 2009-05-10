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

#include <openmbvcppinterface/path.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Path::Path() : Body() {
}

void Path::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Path name=\""<<name<<"\">"<<endl;
    Body::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <color>["<<color[0]<<";"
                                 <<color[1]<<";"
                                 <<color[2]<<"]</color>"<<endl;
  xmlFile<<indent<<"</Path>"<<endl;
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
