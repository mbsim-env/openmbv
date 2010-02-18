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

#ifndef _OPENMBV_IVBODY_H_
#define _OPENMBV_IVBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <string>

namespace OpenMBV {

  /** A body defines by a Open Inventor file or a VRML file */
  class IvBody : public RigidBody {
    public:
      /** Default constructor */
      IvBody();

      /** The file of the iv file to read */
      void setIvFileName(std::string ivFileName_) { ivFileName=ivFileName_; }

      /** Set the limit crease angle for drawing crease edges. 
       * If less 0 do not draw crease edges. Default: -1 */
      void setCreaseEdges(double creaseAngle_) { creaseAngle=creaseAngle_; }

      /** Draw boundary edges or not? Default: false */
      void setBoundaryEdges(bool b) { boundaryEdges=b; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
    protected:
      std::string ivFileName;
      double creaseAngle;
      bool boundaryEdges;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
  };

}

#endif
