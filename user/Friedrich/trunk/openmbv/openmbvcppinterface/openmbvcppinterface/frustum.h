#ifndef _OPENMBV_FRUSTUM_H_
#define _OPENMBV_FRUSTUM_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  class Frustum : public RigidBody {
    protected:
      double baseRadius, topRadius, height, innerBaseRadius, innerTopRadius;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Frustum();
      void setBaseRadius(double radius) {
        baseRadius=radius;
      } 
      void setTopRadius(double radius) {
        topRadius=radius;
      } 
      void setHeight(double height_) {
        height=height_;
      } 
      void setInnerBaseRadius(double radius) {
        innerBaseRadius=radius;
      } 
      void setInnerTopRadius(double radius) {
        innerTopRadius=radius;
      } 
  };

}

#endif
