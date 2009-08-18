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
  if(!hdf5LinkBody && data) delete data;
}

void NurbsDisk::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<NurbsDisk name=\""<<name<<"\">"<<endl;
  DynamicColoredBody::writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"  <scaleFactor>"<<scaleFactor<<"</scaleFactor>"<<endl;
  xmlFile<<indent<<"  <drawDegree>"<<drawDegree<<"</drawDegree>"<<endl;  
  xmlFile<<indent<<"  <innerRadius>"<<Ri<<"</innerRadius>"<<endl;
  xmlFile<<indent<<"  <outerRadius>"<<Ro<<"</outerRadius>"<<endl;
  xmlFile<<indent<<"  <knotVecAzimuthal>[";
  for(int i=0;i<ElementNumberAzimuthal +1+2*InterpolationDegreeAzimuthal;i++) { xmlFile << KnotVecAzimuthal[i] <<";"; }
  xmlFile << "]</knotVecAzimuthal>"  <<endl;
  xmlFile<<indent<<"  <knotVecRadial>[";
  for(int i=0;i<ElementNumberRadial+1+InterpolationDegreeRadial+1;i++) { xmlFile << KnotVecRadial[i] <<";" ; }
  xmlFile << "]</knotVecRadial>"  <<endl;
  xmlFile<<indent<<"  <elementNumberAzimuthal>"<<ElementNumberAzimuthal<<"</elementNumberAzimuthal>"<<endl;
  xmlFile<<indent<<"  <elementNumberRadial>"<<ElementNumberRadial<<"</elementNumberRadial>"<<endl;
  xmlFile<<indent<<"  <interpolationDegreeRadial>"<<InterpolationDegreeRadial<<"</interpolationDegreeRadial>"<<endl;
  xmlFile<<indent<<"  <interpolationDegreeAzimuthal>"<<InterpolationDegreeAzimuthal<<"</interpolationDegreeAzimuthal>"<<endl;
  xmlFile<<indent<<"</NurbsDisk>"<<endl;
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
    for(int i=0;i<ElementNumberAzimuthal*drawDegree*2;i++) {
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

