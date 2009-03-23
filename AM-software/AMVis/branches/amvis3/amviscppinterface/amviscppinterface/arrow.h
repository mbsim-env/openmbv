#ifndef _AMVIS_ARROW_H_
#define _AMVIS_ARROW_H_

#include <amviscppinterface/body.h>
#include <hdf5serie/vectorserie.h>

namespace AMVis {

  class Arrow : public Body {
    protected:
      enum Type {
        noHead,
        fromHead,
        toHead,
        bothHeads
      };
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      double headDiameter, headLength, diameter;
      Type type;
    public:
      Arrow();
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        data->append(row);
      }
      void setArrowHead(float diameter, float length) {
        headDiameter=diameter;
        headLength=length;
      }
      void setDiameter(float diameter_) {
        diameter=diameter_;
      }
      void setType(Type type_) {
        type=type_;
      }
  };

}

#endif
