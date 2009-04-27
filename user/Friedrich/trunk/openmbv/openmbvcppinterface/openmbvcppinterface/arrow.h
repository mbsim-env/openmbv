#ifndef _OPENMBV_ARROW_H_
#define _OPENMBV_ARROW_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** A arrow with zero, one or two heads */
  class Arrow : public Body {
    protected:
      enum Type {
        line,
        fromHead,
        toHead,
        bothHeads
      };
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      double headDiameter, headLength, diameter, scaleLength;
      Type type;
    public:
      /** Default Constructor */
      Arrow();
      
      /** Append the data \p row to the end of the dataset */
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        data->append(row);
      }

      /** Set the \p diameter and \p length of the arrow head (which is a cone) */
      void setArrowHead(float diameter, float length) {
        headDiameter=diameter;
        headLength=length;
      }

      /** Set the \p diameter_ of the arrow (which is a cylinder) */
      void setDiameter(float diameter_) {
        diameter=diameter_;
      }
      
      /** Set the type of the arrow.
       * Use "line" to draw the arrow as a simple line;
       * Use "fromHead" to draw the arrow with a head at the 'from' point;
       * Use "toHead" to draw the arrow with a head at the 'to' point;
       * Use "bothHeads" to draw the arrow with a head at the 'from' and 'to' point;
       */
      void setType(Type type_) {
        type=type_;
      }

      /** Scale the length of the arrow by \p scale */
      void setScaleLength(double scale) {
        scaleLength=scale;
      }
  };

}

#endif
