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

NurbsDisk::NurbsDisk() : DynamicColoredBody(),
  data(0), 
  scaleFactor(1),
  drawDegree(1),
  Ri(0.),
  Ro(0.),
  KnotVecAzimuthal(0),
  KnotVecRadial(0),
  ElementNumberAzimuthal(0),
  ElementNumberRadial(0),
  InterpolationDegreeRadial(3),
  InterpolationDegreeAzimuthal(8),
  DiskNormal(0),
  DiskPoint(0) {
  }

NurbsDisk::~NurbsDisk() {
  if(!hdf5LinkBody && data) { delete data; data=0; }
  if(KnotVecAzimuthal) { delete[] KnotVecAzimuthal; KnotVecAzimuthal=0; }
  if(KnotVecRadial) { delete[] KnotVecRadial; KnotVecRadial=0; }
}

TiXmlElement *NurbsDisk::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=DynamicColoredBody::writeXMLFile(parent);
  addElementText(e, "scaleFactor", scaleFactor);
  addElementText(e, "drawDegree", drawDegree);
  addElementText(e, "innerRadius", Ri);
  addElementText(e, "outerRadius", Ro);
  string str="[";
  for(int i=0;i<ElementNumberAzimuthal +1+2*InterpolationDegreeAzimuthal;i++) str+=numtostr(KnotVecAzimuthal[i])+";";
  addElementText(e, "knotVecAzimuthal", str+"]");
  str="[";
  for(int i=0;i<ElementNumberRadial+1+InterpolationDegreeRadial+1;i++) str+=numtostr(KnotVecRadial[i])+";";
  addElementText(e, "knotVecRadial", str+"]");
  addElementText(e, "elementNumberAzimuthal", ElementNumberAzimuthal);
  addElementText(e, "elementNumberRadial", ElementNumberRadial);
  addElementText(e, "interpolationDegreeRadial", InterpolationDegreeRadial);
  addElementText(e, "InterpolationDegreeAzimuthal", InterpolationDegreeAzimuthal);
  return 0;
}

void NurbsDisk::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    int NodeDofs;
    NodeDofs = (ElementNumberRadial + 1) * (ElementNumberAzimuthal + InterpolationDegreeAzimuthal); 
    columns.push_back("Time");
    for(int i=0;i<NodeDofs;i++) { 
      columns.push_back("x"+numtostr(i));
      columns.push_back("y"+numtostr(i));
      columns.push_back("z"+numtostr(i));
    }
    for(int i=0;i<ElementNumberAzimuthal*get(drawDegree)*2;i++) {
      columns.push_back("x"+numtostr(i+NodeDofs));
      columns.push_back("y"+numtostr(i+NodeDofs));
      columns.push_back("z"+numtostr(i+NodeDofs));    
    }
    columns.push_back("Pos x");
    columns.push_back("Pos y");
    columns.push_back("Pos z"); 
    for(int i=0;i<3;i++) {
      for(int j=0;j<3;j++) {
        columns.push_back("Rot "+numtostr(i)+numtostr(j));
      }
    }

    data->create(*hdf5Group,"data",columns);
  }
}

void NurbsDisk::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    data->open(*hdf5Group,"data");
  }
}
