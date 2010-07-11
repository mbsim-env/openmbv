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

#ifndef _OPENMBV_CUBE_H_
#define _OPENMBV_CUBE_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A cube */
  class Cube : public RigidBody {
    protected:
      ScalarParameter length;
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
    public:
      /** Default constructor */
      Cube();

      /** Retrun the class name */
      std::string getClassName() { return "Cube"; }

      /** Set the length of the cube (x, y and z)*/
      void setLength(ScalarParameter length_) {
        set(length,length_);
      } 

      double getLength() { return get(length); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
