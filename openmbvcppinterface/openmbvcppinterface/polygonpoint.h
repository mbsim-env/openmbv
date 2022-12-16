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

#ifndef _OPENMBV_POLYGONPOINT_H_
#define _OPENMBV_POLYGONPOINT_H_

#include <vector>
#include <fstream>
#include <iostream>
#include "openmbvcppinterface/body.h"
#include <stdexcept>

namespace OpenMBV {

  /*!
   * Polygon point
   * x and y are the coordinates of a polygon-edge. If b is 0 this
   * edge is in reality not a edge and is rendered smooth in OpenMBV. If b is 1
   * this edge is rendered non-smooth in OpenMBV.
   */
  class PolygonPoint {
    protected:
      PolygonPoint(double x_, double y_, int b_) : x(x_), y(y_), b(b_) {}
      ~PolygonPoint() = default;
      static void deleter(PolygonPoint *pp) { delete pp; }
    public:
      static std::shared_ptr<PolygonPoint> create(double x_, double y_, int b_) {
        return {new PolygonPoint(x_, y_, b_), &deleter};
      };

      /* GETTER / SETTER */
      double getXComponent() { return x; } 
      double getYComponent() { return y; } 
      int getBorderValue() { return b; }
      /***************************************************/

      /* CONVENIENCE */
      /** write vector of polygon points to XML file */
      static void serializePolygonPointContour(xercesc::DOMElement *parent,
        const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &cont);

      static std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > initializeUsingXML(xercesc::DOMElement *element);

    private:
      double x, y;
      int b;
  };

}

#endif /* POLYGONPOINT_H */

