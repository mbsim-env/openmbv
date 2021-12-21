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

#ifndef _OPENMBV_NURBSSURFACE_H
#define _OPENMBV_NURBSSURFACE_H

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A nurbs surface */
  class NurbsSurface : public RigidBody {
    friend class ObjectFactory;
    protected:
      std::vector<std::vector<double> > cp;
      int numU{0}, numV{0};
      std::vector<double> uKnot, vKnot;
      NurbsSurface() = default;
      ~NurbsSurface() override = default;
    public:
      /** Get control points
       */
      const std::vector<std::vector<double> >& getControlPoints() { return cp; }
      int getNumberOfUControlPoints() { return numU; }
      int getNumberOfVControlPoints() { return numV; }
      const std::vector<double>& getUKnotVector() { return uKnot; }
      const std::vector<double>& getVKnotVector() { return vKnot; }

      /** Set control points
       */
      void setControlPoints(const std::vector<std::vector<double> > &cp_) { cp = cp_; }
      void setNumberOfUControlPoints(int numU_) { numU = numU_; }
      void setNumberOfVControlPoints(int numV_) { numV = numV_; }
      void setUKnotVector(const std::vector<double>& uKnot_) { uKnot = uKnot_; }
      void setVKnotVector(const std::vector<double>& vKnot_) { vKnot = vKnot_; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
