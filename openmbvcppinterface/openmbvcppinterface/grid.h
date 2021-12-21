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

#ifndef _OPENMBV_GRID_H_
#define _OPENMBV_GRID_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A grid in x-y-Plane */
  class Grid : public RigidBody {
    friend class ObjectFactory;
    protected:
      double xSize{1};
      double ySize{1};
      unsigned int nx{10};
      unsigned int ny{10};
      Grid();
      ~Grid() override = default;
    public:
      /** Set the length in x-direction*/
      void setXSize(double length_) {
        xSize=length_;
      } 

      double getXSize() { return xSize; }

      /** Set the length in y-direction*/
      void setYSize(double length_) {
        ySize=length_;
      } 

      double getYSize() { return ySize; }

      /** Set the number of lines in x-direction*/
      void setXNumber(unsigned int n_) {
        nx=n_;
      }

      unsigned int getXNumber() { return nx; }

      /** Set the number of lines in x-direction*/
      void setYNumber(unsigned int n_) {
        ny=n_;
      }
      unsigned int getYNumber() { return ny; }


      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
