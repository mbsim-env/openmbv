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
      double startAngle, endAngle;
      std::vector<PolygonPoint*> *contour;
      Rotation();
      ~Rotation();
    public:
      /** Retrun the class name */
      std::string getClassName() { return "Rotation"; }

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
      void setContour(std::vector<PolygonPoint*> *contour_) {
        contour=contour_;
      }

      std::vector<PolygonPoint*>* getContour() { return contour; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);

  };

}

#endif
