#ifndef _OPENMBV_RIGIDBODY_H_
#define _OPENMBV_RIGIDBODY_H_

#include <openmbvcppinterface/body.h>
#include <vector>
#include <assert.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** Abstract base class for all rigid bodies */
  class RigidBody : public Body {
    friend class CompoundRigidBody;
    protected:
      double minimalColorValue, maximalColorValue;
      std::vector<double> initialTranslation;
      std::vector<double> initialRotation;
      double scaleFactor;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      double staticColor;
    public:
      /** Default constructor */
      RigidBody();

      ~RigidBody();

      /** Set the minimal color value.
       * The color value of the body in linearly mapped between minimalColorValue
       * and maximalColorValue to blue(minimal) over cyan, green, yellow to red(maximal).
       */
      void setMinimalColorValue(const double min) {
        minimalColorValue=min;
      }

      /** Set the maximal color value.
       * See also minimalColorValue
       */
      void setMaximalColorValue(const double max) {
        maximalColorValue=max;
      }

      /** Set initial translaton between the local frame of the body and the reference frame */
      void setInitialTranslation(const std::vector<double>& initTrans) {
        assert(initTrans.size()==3);
        initialTranslation=initTrans;
      }

      /** Set initial translaton between the local frame of the body and the reference frame */
      void setInitialTranslation(double x, double y, double z) {
        std::vector<double> initTrans;
        initTrans.push_back(x);
        initTrans.push_back(y);
        initTrans.push_back(z);
        initialTranslation=initTrans;
      }

      /** Set initial rotation between the local frame of the body and the reference frame.
       * Use cardan angles to represent the transformation matrix
       */
      void setInitialRotation(const std::vector<double>& initRot) {
        assert(initRot.size()==3);
        initialRotation=initRot;
      }

      /** Set initial rotation between the local frame of the body and the reference frame.
       * Use cardan angles to represent the transformation matrix
       */
      void setInitialRotation(double a, double b, double g) {
        std::vector<double> initRot;
        initRot.push_back(a);
        initRot.push_back(b);
        initRot.push_back(g);
        initialRotation=initRot;
      }

      /** Set the scale factor of the body */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      /** Set a static color for the body.
       * If this value is set, the color given to the append function
       * (as last element of the data row) is overwritten with this value.
       */
      void setStaticColor(const double col) {
        staticColor=col;
      }

      /** Append a data vector the the h5 datsset */
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        if(staticColor>=0) row[7]=staticColor;
        data->append(row);
      }
  };

}

#endif
