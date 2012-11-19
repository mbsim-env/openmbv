#ifndef _INDEXEDTESSELATIONFACE_H_
#define _INDEXEDTESSELATIONFACE_H_

#include <Inventor/nodes/SoGroup.h>
#include <Inventor/fields/SoMFInt32.h>
#include <Inventor/fields/SoMFVec3d.h>
#include <Inventor/fields/SoSFEnum.h>

namespace OpenMBVGUI {

class IndexedTesselationFace : public SoGroup {
 SO_NODE_HEADER(IndexedTesselationFace);
 public:
   SoSFEnum windingRule;
   SoMFVec3d coordinate;
   SoMFInt32 coordIndex;

   static void initClass();
   IndexedTesselationFace();
   IndexedTesselationFace(int numChilderen);
   enum WindingRule { ODD, NONZERO, POSITIVE, NEGATIVE, ABS_GEQ_TWO };

   void write(SoWriteAction *action);

   // This function must be called after all attributes are set.
   // When reading from a file it is automatically called.
   // This is a HACK because IndexedTesselationFace is not clearly implemented.
   void generate() { readChildren(NULL); }

 protected:
   SbBool readChildren(SoInput *in);

 private:
   virtual ~IndexedTesselationFace();
   void constructor();
};

}

#endif
