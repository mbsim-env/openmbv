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
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      DoubleParam springRadius, crossSectionRadius, scaleFactor, numberOfCoils;
    public:
      /** Default Constructor */
      CoilSpring();
      
      /** Destructor */
      virtual ~CoilSpring();
      
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        if(!std::isnan((double)dynamicColor)) row[7]=dynamicColor;
        data->append(row);
      }
      void setSpringRadius(DoubleParam radius) { springRadius=radius; }
      void setCrossSectionRadius(DoubleParam radius) { crossSectionRadius=radius; }
      void setScaleFactor(DoubleParam scale) { scaleFactor=scale; }
      void setNumberOfCoils(DoubleParam nr) { numberOfCoils=nr; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
