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
