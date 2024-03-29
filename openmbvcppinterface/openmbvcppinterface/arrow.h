/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef _OPENMBV_ARROW_H_
#define _OPENMBV_ARROW_H_

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <hdf5serie/vectorserie.h>
#include <stdexcept>

namespace OpenMBV {

  /** A arrow with zero, one or two heads
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order given in
   * world frame: time,
   * "to" point x, "to" point y,
   * "to" point z, delta x, delta y, delta z, color */
  class Arrow : public DynamicColoredBody {
    friend class ObjectFactory;
    public:
      enum Type {
        line,
        fromHead,
        toHead,
        bothHeads,
        fromDoubleHead,
        toDoubleHead,
        bothDoubleHeads
      };
      enum ReferencePoint {
        toPoint,
        fromPoint,
        midPoint
      };
    protected:
      std::string pathStr;
      void createHDF5File() override;
      void openHDF5File() override;
      H5::VectorSerie<double>* data{nullptr};
      double headDiameter{0.5};
      double headLength{0.75};
      double diameter{0.25};
      double scaleLength{1};
      Type type{toHead};
      ReferencePoint referencePoint{toPoint};

      Arrow();
      ~Arrow() override;
    public:
      /** Draw path of this object in the viewer if true (the default) */
      void setPath(bool p) { pathStr=(p)?"true":"false"; }

      bool getPath() { return pathStr=="true"?true:false; }
      
      /** Append the data \p row to the end of the dataset */
      template<typename T>
      void append(const T& row) {
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object");
        if(row.size()!=8) throw std::runtime_error("the dimension does not match");
        if(!std::isnan(dynamicColor))
        {
          std::vector<double> tmprow(8);
          std::copy(&row[0], &row[8], tmprow.begin());
          tmprow[7]=dynamicColor;
          data->append(tmprow);
        }
        else
          data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override { return data?data->getRow(i):std::vector<double>(8); }

      /** Convenience; see setHeadDiameter and setHeadLength */
      void setArrowHead(double diameter, double length) {
        headDiameter=diameter;
        headLength=length;
      }

      /** Set the diameter of the arrow head (which is a cone) */
      void setHeadDiameter(double diameter) {
        headDiameter=diameter;
      }

      double getHeadDiameter() { return headDiameter; }

      /** Set the length of the arrow head (which is a cone) */
      void setHeadLength(double length) {
        headLength=length;
      }

      double getHeadLength() { return headLength; }

      /** Set the diameter of the arrow (which is a cylinder) */
      void setDiameter(double diameter_) {
        diameter=diameter_;
      }

      double getDiameter() { return diameter; }
      
      /** Set the type of the arrow.
       * Use "line" to draw the arrow as a simple line;
       * Use "fromHead" to draw the arrow with a head at the 'from' point;
       * Use "toHead" to draw the arrow with a head at the 'to' point;
       * Use "bothHeads" to draw the arrow with a head at the 'from' and 'to' point;
       * Use "fromDoubleHead" to draw the arrow with a double head at the 'from' point;
       * Use "toDoubleHead" to draw the arrow with a double head at the 'to' point;
       * Use "bothDoubleHeads" to draw the arrow with a double head at the 'from' and 'to' point;
       */
      void setType(Type type_) {
        type=type_;
      }
      
      Type getType() { return type; }
      
      /** Set the reference point of the arrow.
       * The reference point is the point being stored in the HDF5 file.
       * Use "toPoint" (the default) if the 'to' point is store in the HDF5 file;
       * Use "fromPoint" if the 'from' point is store in the HDF5 file;
       * Use "midPoint" if the 'mid' point is store in the HDF5 file;
       */
      void setReferencePoint(ReferencePoint referencePoint_) {
        referencePoint=referencePoint_;
      }
      
      ReferencePoint getReferencePoint() { return referencePoint; }

      /** Scale the length of the arrow */
      void setScaleLength(double scale) {
        scaleLength=scale;
      }

      double getScaleLength() { return scaleLength; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
