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

#ifndef _OPENMBV_NURBSCURVE_H
#define _OPENMBV_NURBSCURVE_H

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A nurbs curve */
  class NurbsCurve : public RigidBody {
    friend class ObjectFactory;
    protected:
      std::vector<std::vector<double> > cp;
      int num{0};
      std::vector<double> knot;
      NurbsCurve() = default;
      ~NurbsCurve() override = default;
    public:
      /** Get control points
       */
      const std::vector<std::vector<double> >& getControlPoints() { return cp; }
      int getNumberOfControlPoints() { return num; }
      const std::vector<double>& getKnotVector() { return knot; }

      /** Set control points
       */
      void setControlPoints(const std::vector<std::vector<double> > &cp_) { cp = cp_; }
      void setNumberOfControlPoints(int num_) { num = num_; }
      void setKnotVector(const std::vector<double>& knot_) { knot = knot_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
