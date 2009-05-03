/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

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

      /** Convenience; see setHeadDiameter and setHeadLength */
      void setArrowHead(float diameter, float length) {
        headDiameter=diameter;
        headLength=length;
      }

      /** Set the diameter of the arrow head (which is a cone) */
      void setHeadDiameter(float diameter) {
        headDiameter=diameter;
      }

      /** Set the length of the arrow head (which is a cone) */
      void setHeadLength(float length) {
        headLength=length;
      }

      /** Set the diameter of the arrow (which is a cylinder) */
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

      /** Scale the length of the arrow */
      void setScaleLength(double scale) {
        scaleLength=scale;
      }
  };

}

#endif
