/*
   OpenMBV - Open Multi Body Viewer.
   Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
   */

#include <config.h>
#include "openmbvcppinterface/polygonpoint.h"
#include <fmatvec/toString.h>

using namespace std;
using namespace xercesc;
using namespace MBXMLUtils;

namespace OpenMBV {

void PolygonPoint::serializePolygonPointContour(DOMElement *parent,
  const shared_ptr<vector<shared_ptr<PolygonPoint> > > &cont) {
  string str;
  str="[ ";
  for(auto j=cont->begin(); j!=cont->end(); j++) {
    str+=fmatvec::toString((*j)->getXComponent())+", "+fmatvec::toString((*j)->getYComponent())+", "+fmatvec::toString((*j)->getBorderValue());
    if(j+1!=cont->end()) str+=";    "; else str+=" ]";
  }
  E(parent)->addElementText(OPENMBV%"contour", str);
}

shared_ptr<vector<shared_ptr<PolygonPoint> > > PolygonPoint::initializeUsingXML(DOMElement *element) {
  auto matParam=E(element)->getText<vector<vector<double>>>();
  shared_ptr<vector<shared_ptr<PolygonPoint> > > contour=make_shared<vector<shared_ptr<PolygonPoint> > >();
  for(auto & r : matParam) {
    shared_ptr<PolygonPoint> pp=create(r[0], r[1], (int)(r[2]));
    contour->push_back(pp);
  }
  return contour;
}

}
