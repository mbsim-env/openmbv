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

      /** Retrun the class name */
      std::string getClassName() { return "IvBody"; }

      /** The file of the iv file to read */
      void setIvFileName(std::string ivFileName_) { ivFileName=ivFileName_; }

      std::string getIvFileName() { return ivFileName; }

      /** Set the limit crease angle for the calculation of crease edges. 
       * If less 0 do not calculate crease edges. Default: -1,
       * The crease edges are drawn as outline in OpenMBV. */
      void setCreaseEdges(ScalarParameter creaseAngle_) { set(creaseAngle,creaseAngle_); }

      double getCreaseEdges() { return get(creaseAngle); }

      /** Calculate and draw boundary edges or not? Default: false.
       * The boundary edges are drawn as outline in OpenMBV. */
      void setBoundaryEdges(bool b) { boundaryEdges=b; }

      bool getBoundaryEdges() { return boundaryEdges; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);
    protected:
      ~IvBody() {}
      std::string ivFileName;
      ScalarParameter creaseAngle;
      bool boundaryEdges;
  };

}

#endif
