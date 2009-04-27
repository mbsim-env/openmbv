#ifndef _OPENMBV_PATH_H_
#define _OPENMBV_PATH_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>
#include <vector>

namespace OpenMBV {

  /** Draw a path of a reference point */
  class Path : public Body {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      std::vector<double> color;
    public:
      /** Default constructor */
      Path();

      /** Append a data vector the to hf dataset */
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==4);
        data->append(row);
      }

      /** Set the color of the paht.
       * Use a vector with tree double representing reg, green and blue as paremter.
       * red, green and blue runs form 0 to 1
       */
      void setColor(const std::vector<double>& color_) {
        assert(color_.size()==3);
        color=color_;
      }

      /** Set the color of the paht.
       * red, green and blue runs form 0 to 1
       */
      void setColor(double red, double green, double blue) {
        color.clear();
        color.push_back(red);
        color.push_back(green);
        color.push_back(blue);
      }
  };

}

#endif
