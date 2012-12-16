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

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <hdf5serie/vectorserie.h>
#include <cmath>

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
      void createHDF5File();
      void openHDF5File();
      H5::VectorSerie<double>* data;
      ScalarParameter headDiameter, headLength, diameter, scaleLength;
      Type type;
      ReferencePoint referencePoint;

      /** Destructor */
      virtual ~Arrow();
    public:
      /** Default Constructor */
      Arrow();

      /** Retrun the class name */
      std::string getClassName() { return "Arrow"; }

      /** Draw path of this object in the viewer if true (the default) */
      void setPath(bool p) { pathStr=(p==true)?"true":"false"; }

      bool getPath() { return pathStr=="true"?true:false; }
      
      /** Append the data \p row to the end of the dataset */
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        if(!std::isnan(dynamicColor)) row[7]=dynamicColor;
        data->append(row);
      }

      int getRows() { return data?data->getRows():-1; }
      std::vector<double> getRow(int i) { return data?data->getRow(i):std::vector<double>(8); }

      /** Convenience; see setHeadDiameter and setHeadLength */
      void setArrowHead(ScalarParameter diameter, ScalarParameter length) {
        set(headDiameter,diameter);
        set(headLength,length);
      }

      /** Set the diameter of the arrow head (which is a cone) */
      void setHeadDiameter(ScalarParameter diameter) {
        set(headDiameter,diameter);
      }

      double getHeadDiameter() { return get(headDiameter); }

      /** Set the length of the arrow head (which is a cone) */
      void setHeadLength(ScalarParameter length) {
        set(headLength,length);
      }

      double getHeadLength() { return get(headLength); }

      /** Set the diameter of the arrow (which is a cylinder) */
      void setDiameter(ScalarParameter diameter_) {
        set(diameter,diameter_);
      }

      double getDiameter() { return get(diameter); }
      
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
      void setScaleLength(ScalarParameter scale) {
        set(scaleLength,scale);
      }

      double getScaleLength() { return get(scaleLength); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

      TiXmlElement *writeXMLFile(TiXmlNode *parent);
  };

}

#endif
