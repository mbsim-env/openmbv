#ifndef _OPENMBV_PATH_H_
#define _OPENMBV_PATH_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>
#include <vector>

namespace OpenMBV {

  class Path : public Body {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      std::vector<double> color;
    public:
      Path();
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==4);
        data->append(row);
      }
      void setColor(const std::vector<double>& color_) {
        assert(color_.size()==3);
        color=color_;
      }
      void setColor(double r, double g, double b) {
        color.clear();
        color.push_back(r);
        color.push_back(g);
        color.push_back(b);
      }
  };

}

#endif
