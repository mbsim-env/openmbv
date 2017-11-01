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

#include <config.h>
#include "openmbvcppinterface/polygonpoint.h"

using namespace std;
using namespace xercesc;
using namespace MBXMLUtils;

namespace OpenMBV {

void PolygonPoint::serializePolygonPointContour(DOMElement *parent,
  const shared_ptr<vector<shared_ptr<PolygonPoint> > > &cont) {
  string str;
  str="[ ";
  for(vector<shared_ptr<PolygonPoint> >::const_iterator j=cont->begin(); j!=cont->end(); j++) {
    str+=to_string((*j)->getXComponent())+", "+to_string((*j)->getYComponent())+", "+to_string((*j)->getBorderValue());
    if(j+1!=cont->end()) str+=";    "; else str+=" ]";
  }
  E(parent)->addElementText(OPENMBV%"contour", str);
}

shared_ptr<vector<shared_ptr<PolygonPoint> > > PolygonPoint::initializeUsingXML(DOMElement *element) {
  vector<vector<double> > matParam=E(element)->getText<vector<vector<double>>>();
  shared_ptr<vector<shared_ptr<PolygonPoint> > > contour=make_shared<vector<shared_ptr<PolygonPoint> > >();
  for(size_t r=0; r<matParam.size(); r++) {
    shared_ptr<PolygonPoint> pp=create(matParam[r][0], matParam[r][1], (int)(matParam[r][2]));
    contour->push_back(pp);
  }
  return contour;
}

}
