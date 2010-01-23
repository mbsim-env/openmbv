#ifndef _INDEXEDTESSELATIONFACE_H_
#define _INDEXEDTESSELATIONFACE_H_

#include <Inventor/nodes/SoGroup.h>
#include <Inventor/fields/SoMFInt32.h>
#include <Inventor/fields/SoMFVec3d.h>
#include <Inventor/fields/SoSFEnum.h>
#include <Inventor/sensors/SoNodeSensor.h>

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
   enum WindingRule { ODD, NONZERO, POSITIVE, NEGATIVE, ABS_GEQ_TWO };
   static void changedCB(void *data, SoSensor*);
   friend class Body;

 private:
   virtual ~IndexedTesselationFace();
   void constructor();
};

#endif
