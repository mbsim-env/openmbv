#ifndef _RIGIDBODY_H_
#define _RIGIDBODY_H_

#include <amviscppinterface/body.h>
#include <vector>
#include <assert.h>
#include <hdf5serie/vectorserie.h>

namespace AMVis {

  class RigidBody : public Body {
    protected:
      std::vector<double> initialTranslation;
      std::vector<double> initialRotation;
      double scaleFactor;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
    public:
      RigidBody(const std::string& name_);
      ~RigidBody();
      void setInitialTranslation(const std::vector<double>& initTrans) {
        assert(initTrans.size()==3);
        initialTranslation=initTrans;
      }
      void setInitialRotation(const std::vector<double>& initRot) {
        assert(initRot.size()==3);
        initialRotation=initRot;
      }
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }
      void append(const std::vector<double>& row) {
        data->append(row);
      }
  };

}

#endif
