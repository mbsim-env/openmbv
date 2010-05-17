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
      DoubleParam xSize, ySize;
      unsigned int nx, ny;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      /** Default constructor */
      Grid();

      /** Set the length in x-direction*/
      void setXSize(DoubleParam length_) {
        xSize=length_;
      } 

      /** Set the length in y-direction*/
      void setYSize(DoubleParam length_) {
        ySize=length_;
      } 

      /** Set the number of lines in x-direction*/
      void setXNumber(unsigned int n_) {
        nx=n_;
      }

      /** Set the number of lines in x-direction*/
      void setYNumber(unsigned int n_) {
        ny=n_;
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
