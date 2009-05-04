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

#include <openmbvcppinterface/rotation.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Rotation::Rotation() : RigidBody(),
  contour(0) {
}

void Rotation::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Rotation name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    if(contour) {
      xmlFile<<indent<<"  <contour>"<<endl;
      xmlFile<<indent<<"    [ ";
      for(vector<PolygonPoint*>::const_iterator j=contour->begin(); j!=contour->end(); j++) {
        if(j!=contour->begin()) xmlFile<<indent<<"      ";
        xmlFile<<(*j)->x<<", "<<(*j)->y<<", "<<(*j)->b;
        if(j+1!=contour->end()) xmlFile<<";"<<endl; else xmlFile<<" ]"<<endl;
      }
      xmlFile<<indent<<"  </contour>"<<endl;
    }
  xmlFile<<indent<<"</Rotation>"<<endl;
}
