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

#include <openmbvcppinterface/extrusion.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Extrusion::Extrusion() : RigidBody(),
  windingRule(odd),
  height(1) {
  }

void Extrusion::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Extrusion name=\""<<name<<"\" expand=\""<<expandStr<<"\">"<<endl;
  RigidBody::writeXMLFile(xmlFile, indent+"  ");
  string windingRuleStr;
  switch(windingRule) {
    case odd: windingRuleStr="odd"; break;
    case nonzero: windingRuleStr="nonzero"; break;
    case positive: windingRuleStr="positive"; break;
    case negative: windingRuleStr="negative"; break;
    case absGEqTwo: windingRuleStr="absGEqTwo"; break;
  }
  xmlFile<<indent<<"  <windingRule>"<<windingRuleStr<<"</windingRule>"<<endl;
  xmlFile<<indent<<"  <height>"<<height<<"</height>"<<endl;
  for(vector<vector<PolygonPoint*>*>::const_iterator i=contour.begin(); i!=contour.end(); i++) 
    PolygonPoint::serializePolygonPointContour(xmlFile, indent, (*i));
  xmlFile<<indent<<"</Extrusion>"<<endl;
}

