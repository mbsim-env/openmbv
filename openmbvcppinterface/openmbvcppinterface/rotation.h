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

#ifndef _OPENMBV_ROTATION_H_
#define _OPENMBV_ROTATION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

  /** Rotation of a cross section area */
  class Rotation : public RigidBody {
    friend class ObjectFactory;
    protected:
      double startAngle{0};
      double endAngle;
      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > contour;
      Rotation();
      ~Rotation() override;
    public:
      /** Set start angle of the rotation (Default: 0). */
      void setStartAngle(double angle) {
        startAngle=angle;
      }

      double getStartAngle() { return startAngle; }

      /** Set end angle of the rotation (Default: 2*pi). */
      void setEndAngle(double angle) {
        endAngle=angle;
      }

      double getEndAngle() { return endAngle; }

      /** Set start and end angle of the rotation (Default 0-2*pi). */
      void setAngle(double startAngle_, double endAngle_) {
        startAngle=startAngle_;
        endAngle=endAngle_;
      }

      /** Set cross section area of the rotation.
       * The cross section is rotation around the local y-axis
       */
      void setContour(const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &contour_) {
        contour=contour_;
      }

      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > getContour() { return contour; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

  };

}

#endif
