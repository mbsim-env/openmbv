#ifndef _OPENMBV_COILSPRING_H_
#define _OPENMBV_COILSPRING_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  class CoilSpring : public Body {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      double springRadius, crossSectionRadius, scaleFactor, numberOfCoils;
    public:
      CoilSpring();
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        data->append(row);
      }
      void setSpringRadius(double radius) { springRadius=radius; }
      void setCrossSectionRadius(double radius) { crossSectionRadius=radius; }
      void setScaleFactor(double scale) { scaleFactor=scale; }
      void setNumberOfCoils(double nr) { numberOfCoils=nr; }
  };

}

#endif
