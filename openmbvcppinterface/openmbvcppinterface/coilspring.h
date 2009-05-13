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

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** A coil spring */
  class CoilSpring : public Body {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      double springRadius, crossSectionRadius, scaleFactor, numberOfCoils;
    public:
      CoilSpring();
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        data->append(row);
      }
      void setSpringRadius(double radius) { springRadius=radius; }
      void setCrossSectionRadius(double radius) { crossSectionRadius=radius; }
      void setScaleFactor(double scale) { scaleFactor=scale; }
      void setNumberOfCoils(double nr) { numberOfCoils=nr; }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
