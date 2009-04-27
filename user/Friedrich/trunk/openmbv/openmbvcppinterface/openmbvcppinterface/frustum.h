#ifndef _OPENMBV_FRUSTUM_H_
#define _OPENMBV_FRUSTUM_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A frustum (with a frustum hole) */
  class Frustum : public RigidBody {
    protected:
      double baseRadius, topRadius, height, innerBaseRadius, innerTopRadius;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      /** Default constructor */
      Frustum();

      /** Set the radius of the outer side at the base (bottom) */
      void setBaseRadius(double radius) {
        baseRadius=radius;
      } 

      /** Set the radius of the outer side at the top. */
      void setTopRadius(double radius) {
        topRadius=radius;
      } 

      /** Set height of the frustum */
      void setHeight(double height_) {
        height=height_;
      } 

      /** Set the radius of the inner side at the base (bottom). */
      void setInnerBaseRadius(double radius) {
        innerBaseRadius=radius;
      } 

      /** Set the radius of the inner side at the top. */
      void setInnerTopRadius(double radius) {
        innerTopRadius=radius;
      } 
  };

}

#endif
