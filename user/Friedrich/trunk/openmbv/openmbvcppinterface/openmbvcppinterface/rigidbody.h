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
      std::vector<double> initialTranslation;
      std::vector<double> initialRotation;
      double scaleFactor;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
    public:
      RigidBody();
      ~RigidBody();
      void setInitialTranslation(const std::vector<double>& initTrans) {
        assert(initTrans.size()==3);
        initialTranslation=initTrans;
      }
      void setInitialTranslation(double x, double y, double z) {
        std::vector<double> initTrans;
        initTrans.push_back(x);
        initTrans.push_back(y);
        initTrans.push_back(z);
        initialTranslation=initTrans;
      }
      void setInitialRotation(const std::vector<double>& initRot) {
        assert(initRot.size()==3);
        initialRotation=initRot;
      }
      void setInitialRotation(double a, double b, double g) {
        std::vector<double> initRot;
        initRot.push_back(a);
        initRot.push_back(b);
        initRot.push_back(g);
        initialRotation=initRot;
      }
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        data->append(row);
      }
  };

}

#endif
