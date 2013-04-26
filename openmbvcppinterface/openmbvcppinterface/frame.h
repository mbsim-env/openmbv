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

#ifndef _OPENMBV_FRAME_H
#define _OPENMBV_FRAME_H

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A frame; A coordinate system */
  class Frame : public RigidBody {
    protected:
      ScalarParameter size;
      ScalarParameter offset;
      ~Frame() {}
    public:
      /** Default constructor */
      Frame();

      /** Retrun the class name */
      std::string getClassName() { return "Frame"; }

      /** Set the length of the three axis, represended by lines in red, green and blue color. */
      void setSize(ScalarParameter size_) { set(size,size_); }

      double getSize() { return get(size); }

      /** Set the offset of the thre axis.
       * A offset of 0 means, that the axis/lines are intersecting in there mid points.
       * A offset of 1 menas, that the axis/lines are intersecting at there start points.
       */
      void setOffset(ScalarParameter offset_) { set(offset,offset_); }

      double getOffset() { return get(offset); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(MBXMLUtils::TiXmlElement *element);

      MBXMLUtils::TiXmlElement* writeXMLFile(MBXMLUtils::TiXmlNode *parent);
  };

}

#endif
