#ifndef _AMVIS_CYLINDER_H_
#define _AMVIS_CYLINDER_H_

#include <amviscppinterface/rigidbody.h>

namespace AMVis {

  class Cylinder : public RigidBody {
    protected:
      double baseRadius, topRadius, height, innerBaseRadius, innerTopRadius;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Cylinder();
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
