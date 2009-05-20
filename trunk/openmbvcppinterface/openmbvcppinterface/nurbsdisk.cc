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

#include <openmbvcppinterface/nurbsdisk.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

NurbsDisk::NurbsDisk() : Body(),
  data(0), 
  staticColor(-1),
  minimalColorValue(0.),
  maximalColorValue(1.),
  scaleFactor(1), 
  Ri(0.),
  Ro(0.),
  KnotVecAzimuthal(0),
  KnotVecRadial(0),
  ElementNumberAzimuthal(0),
  ElementNumberRadial(0),
  InterpolationDegree(3) {
    // TODO thickness 
  }

  NurbsDisk::~NurbsDisk() {
    if(!hdf5LinkBody && data) delete data;
  }

void NurbsDisk::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<NurbsDisk name=\""<<name<<"\">"<<endl;
  Body::writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"  <minimalColorValue>"<<minimalColorValue<<"</minimalColorValue>"<<endl;
  xmlFile<<indent<<"  <maximalColorValue>"<<maximalColorValue<<"</maximalColorValue>"<<endl;
  xmlFile<<indent<<"  <scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
  xmlFile<<indent<<"  <color>"<<staticColor<<"</color>"<<endl;
  xmlFile<<indent<<"  <ElementNumberAzimuthal>"<<ElementNumberAzimuthal<<"</ElementNumberAzimuthal>"<<endl;
  xmlFile<<indent<<"  <ElementNumberRadial>"<<ElementNumberRadial<<"</ElementNumberRadial>"<<endl;
  xmlFile<<indent<<"  <KnotVecAzimuthal>"<<KnotVecAzimuthal<<"</KnotVecAzimuthal>"<<endl;
  xmlFile<<indent<<"  <KnotVecRadial>"<<KnotVecRadial<<"</KnotVecRadial>"<<endl;
  xmlFile<<indent<<"</NurbsDisk>"<<endl;
}

void NurbsDisk::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    columns.push_back("Time");
    for(int i=0;i<22;i++) { //NodeDofs
      columns.push_back("x"+numtostr(i));
      columns.push_back("y"+numtostr(i));
      columns.push_back("z"+numtostr(i));
    }
    data->create(*hdf5Group,"data",columns);
  }
}

