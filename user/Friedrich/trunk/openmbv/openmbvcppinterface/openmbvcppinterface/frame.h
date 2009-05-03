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
      double size;
      double offset;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      /** Default constructor */
      Frame();

      /** Set the length of the three axis, represended by lines in red, green and blue color. */
      void setSize(double size_) { size=size_; }

      /** Set the offset of the thre axis.
       * A offset of 0 means, that the axis/lines are intersecting in there mid points.
       * A offset of 1 menas, that the axis/lines are intersecting at there start points.
       */
      void setOffset(double offset_) { offset=offset_; }
  };

}

#endif
