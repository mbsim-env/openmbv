#ifndef _INDEXEDTESSELATIONFACE_H_
#define _INDEXEDTESSELATIONFACE_H_

#include <Inventor/nodes/SoGroup.h>
#include <Inventor/fields/SoMFInt32.h>
#include <Inventor/fields/SoMFVec3d.h>
#include <Inventor/fields/SoSFEnum.h>

class IndexedTesselationFace : public SoGroup {
 SO_NODE_HEADER(IndexedTesselationFace);
 public:
   SoSFEnum windingRule;
   SoMFVec3d coordinate;
   SoMFInt32 coordIndex;

   static void initClass();
   IndexedTesselationFace();
   IndexedTesselationFace(int numChilderen);

 protected:
   SbBool readChildren(SoInput *in);
   enum WindingRule { ODD, NONZERO, POSITIVE, NEGATIVE, ABS_GEQ_TWO };
   friend class Body;

 private:
   virtual ~IndexedTesselationFace();
   void constructor();
};

#endif
