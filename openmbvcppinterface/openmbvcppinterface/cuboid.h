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

#ifndef _OPENMBV_CUBOID_H_
#define _OPENMBV_CUBOID_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A cuboid */
  class Cuboid : public RigidBody {
    friend class ObjectFactory;
    protected:
      std::vector<double> length;
      Cuboid();
      ~Cuboid() override = default;
    public:
      /** Set the length of the cuboid */
      void setLength(const std::vector<double>& length_) {
        if(length_.size()!=3) throw std::runtime_error("the dimension does not match");
        length=length_;
      } 

      std::vector<double> getLength() { return length; }

      /** Set the length of the cuboid */
      void setLength(double x, double y, double z) {
        std::vector<double> length_;
        length_.push_back(x);
        length_.push_back(y);
        length_.push_back(z);
        length=length_;
      } 

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
