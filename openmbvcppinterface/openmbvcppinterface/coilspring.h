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

#ifndef _OPENMBV_COILSPRING_H_
#define _OPENMBV_COILSPRING_H_

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <hdf5serie/vectorserie.h>
#include <stdexcept>

namespace OpenMBV {

  /** A coil spring
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order given in
   * world frame: time,
   * "from" point x, "from" point y,
   * "from" point z, "to" point x, "to" point y, "to" point z, color */
  class CoilSpring : public DynamicColoredBody {
    friend class ObjectFactory;
    public:
      enum Type {
        tube,
        scaledTube,
        polyline
      };
    protected:
      void createHDF5File() override;
      void openHDF5File() override;
      H5::VectorSerie<double>* data{nullptr};
      double springRadius{1};
      double crossSectionRadius{-1};
      double scaleFactor{1};
      double numberOfCoils{3};
      double nominalLength{-1};
      Type type{tube};
      
      CoilSpring();
      ~CoilSpring() override;
    public:
      template<typename T>
      void append(const T& row) {
        if(data==nullptr) throw std::runtime_error("can not append data to an environement object");
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

      void setSpringRadius(double radius) { springRadius=radius; }
      double getSpringRadius() { return springRadius; }

      /** The radius of the coil spring cross-section if type=tube or type=scaledTube.
       * If type=polyline this parameter defines the point size of the polyline.
       * If crossSectionRadius is less then 0, the cross-section radius
       * is choosen automatically.
       */
      void setCrossSectionRadius(double radius) { crossSectionRadius=radius; }
      double getCrossSectionRadius() { return crossSectionRadius; }

      void setScaleFactor(double scale) { scaleFactor=scale; }
      double getScaleFactor() { return scaleFactor; }

      void setNumberOfCoils(double nr) { numberOfCoils=nr; }
      double getNumberOfCoils() { return numberOfCoils; }

      /** Set the nominal length of the coil spring.
       * This parameter is only usefull, if type=scaledTube: in this case
       * the cross-section of the coil is an exact circle if the spring
       * length is nominalLength. In all other cases the cross-section
       * scales with the spring length and is getting a ellipse.
       * If nominalLength is less than 0, the nominalLength is
       * choosen automatically.
       */
      void setNominalLength(double l) { nominalLength=l; }
      double getNominalLength() { return nominalLength; }

      /** The type of the coil spring.
       * "tube": The coil spring geometry is an extrusion of a circle along
       * the spring center line;
       * "scaledTube": The coil spring geometry is an extrusion of a circle along
       * the spring center line with a spring length of nominalLength. This
       * geometry is scaled as a whole for each other spring length. This type is much faster
       * than "tube";
       * "polyline": The coil spring geometry is a polyline representing the
       * the spring center line. this is the faster spring visualisation;
       */
      void setType(Type t) { type=t; }
      Type getType() { return type; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
