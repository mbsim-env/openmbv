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
#include <cmath>

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
    public:
      enum Type {
        tube,
        scaledTube,
        polyline
      };
    protected:
      TiXmlElement *writeXMLFile(TiXmlNode *parent);
      void createHDF5File();
      void openHDF5File();
      H5::VectorSerie<double>* data;
      ScalarParameter springRadius, crossSectionRadius, scaleFactor, numberOfCoils, nominalLength;
      Type type;
    public:
      /** Default Constructor */
      CoilSpring();
      
      /** Destructor */
      virtual ~CoilSpring();

      /** Retrun the class name */
      std::string getClassName() { return "CoilSpring"; }
      
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        if(!std::isnan(dynamicColor)) row[7]=dynamicColor;
        data->append(row);
      }

      int getRows() { return data?data->getRows():-1; }
      std::vector<double> getRow(int i) { return data->getRow(i); }

      void setSpringRadius(ScalarParameter radius) { set(springRadius,radius); }
      double getSpringRadius() { return get(springRadius); }

      /** The radius of the coil spring cross-section if type=tube or type=scaledTube.
       * If type=polyline this parameter defines the point size of the polyline.
       * If crossSectionRadius is less then 0, the cross-section radius
       * is choosen automatically.
       */
      void setCrossSectionRadius(ScalarParameter radius) { set(crossSectionRadius,radius); }
      double getCrossSectionRadius() { return get(crossSectionRadius); }

      void setScaleFactor(ScalarParameter scale) { set(scaleFactor,scale); }
      double getScaleFactor() { return get(scaleFactor); }

      void setNumberOfCoils(ScalarParameter nr) { set(numberOfCoils,nr); }
      double getNumberOfCoils() { return get(numberOfCoils); }

      /** Set the nominal length of the coil spring.
       * This parameter is only usefull, if type=scaledTube: in this case
       * the cross-section of the coil is an exact circle if the spring
       * length is nominalLength. In all other cases the cross-section
       * scales with the spring length and is getting a ellipse.
       * If nominalLength is less than 0, the nominalLength is
       * choosen automatically.
       */
      void setNominalLength(ScalarParameter l) { set(nominalLength,l); }
      double getNominalLength() { return get(nominalLength); }

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
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
