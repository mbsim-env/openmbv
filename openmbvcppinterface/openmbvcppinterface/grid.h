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

#ifndef _OPENMBV_GRID_H_
#define _OPENMBV_GRID_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A grid in x-y-Plane */
  class Grid : public RigidBody {
    protected:
      double xSize, ySize;
      unsigned int nx, ny;
      ~Grid() {}
    public:
      /** Default constructor */
      Grid();

      /** Retrun the class name */
      std::string getClassName() { return "Grid"; }

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
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);
  };

}

#endif
