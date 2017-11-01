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

#ifndef _OPENMBV_EXTRUSION_H_
#define _OPENMBV_EXTRUSION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

  /** A extrusion of a cross section area (with holes) */
  class Extrusion : public RigidBody {
    friend class ObjectFactory;
    public:
      enum WindingRule {
        odd,
        nonzero,
        positive,
        negative,
        absGEqTwo
      };
    protected:
      WindingRule windingRule;
      double height;
      std::vector<std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > > contour;
      Extrusion();
      ~Extrusion();
    public:
      /** Set the OpenGL winding rule for the tesselation of the cross section area.
       * Allowable values are "odd", "nonzero", "positive", "negative" and "absGEqTwo".
       * See the OpenGL-GLU documentation the the meaning of this values.
       */
      void setWindingRule(WindingRule windingRule_) {
        windingRule=windingRule_;
      }

      WindingRule getWindingRule() { return windingRule; }

      /** Set the height of the extrusion.
       * The extrusion is along the normal of the cross section area (local z-axis).
       */
      void setHeight(double height_) {
        height=height_;
      }
      
      double getHeight() { return height; }

      /** Clear all previously added contours. */
      void clearContours() {
        contour.clear();
      }

      /** Add a new contour to the extrusion.
       * See setWindingRule of details about how they are combined.
       */
      void addContour(const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &contour_) {
        contour.push_back(contour_);
      }

      std::vector<std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > >& getContours() {
        return contour;
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent);

  };

}

#endif /* _OPENMBV_EXTRUSION_H_ */

