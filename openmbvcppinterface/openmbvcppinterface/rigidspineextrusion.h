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

#ifndef _OPENMBV_RIGIDSPINEEXTRUSION_H_
#define _OPENMBV_RIGIDSPINEEXTRUSION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>

namespace OpenMBV {

  /** 
   * \brief Class for all rigid bodies extruded along a curve. 
   *
   * The cross section is
   * freely rotated with respect to the world system.
   * The cross section is defined in the x-z-plane
   *
   */
  class RigidSpineExtrusion : public RigidBody {
    friend class ObjectFactory;
    public:
      /** Set the 2D contour (cross-section) of the extrusion.
       * The contour (polygon) points must be in clockwise order. */
      void setContour(const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &contour_) { contour = contour_; }

      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > getContour() { return contour; }

      void setCounterClockWise(bool f) { ccw = f; }
      bool getCounterClockWise() { return ccw; }

      struct Spine {
        Spine(float x_, float y_, float z_, float a, float b, float g) : x(x_), y(y_), z(z_), alpha(a), beta(b), gamma(g) {}
        float x, y, z;
        float alpha, beta, gamma;
      };
      void setSpine(const std::vector<Spine>& spine_) { spine = spine_; };
      std::vector<Spine> getSpine() { return spine; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      /** Write XML file for not time-dependent data. */
      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
    protected:
      RigidSpineExtrusion();
      ~RigidSpineExtrusion() override;

      std::vector<Spine> spine;

      /** Vector of local x-y points. */
      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > contour;

      bool ccw { true };
  };

}

#endif /* _OPENMBV_RIGIDSPINEEXTRUSION_H_ */

