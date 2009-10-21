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

#ifndef _OPENMBV_SPINEEXTRUSION_H_
#define _OPENMBV_SPINEEXTRUSION_H_

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>
#include <assert.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** 
   * \brief Class for all bodies extruded along a curve. 
   * \author Thorsten Schindler
   */
  class SpineExtrusion : public DynamicColoredBody {
    public:
      /** constructor */
      SpineExtrusion();

      /** destructor */
      virtual ~SpineExtrusion();

      /** Set the number of spine points used for extrusion along a path.
       */
      void setNumberOfSpinePoints(const int num) {
        numberOfSpinePoints=num;
      }

      /** Get the number of spine points used for extrusion along a path.
       */
      int getNumberOfSpinePoints() {
        return numberOfSpinePoints;
      }

      /** Set a new contour to the extrusion. */
      void setContour(std::vector<PolygonPoint*> *contour_) { contour = contour_; }

      /** Set the scale factor of the body. */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      /** Append a data vector to the h5 datsset */
      void append(std::vector<double>& row) { 
        assert(data!=0); 
        data->append(row);
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

    protected:
      /** Number of spine points used for extrusion along a path. */
      int numberOfSpinePoints;

      /** Vector of local x-y points. */
      std::vector<PolygonPoint*> *contour;

      /** Each row comprises [time,spine world position,spine twist,...,spine world position,spine twist]. */
      H5::VectorSerie<double>* data;

      /** Scale factor of the body. */
      double scaleFactor;

      /** Write XML file for not time-dependent data. */
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      
      /** Write H5 file for time-dependent data. */
      void createHDF5File();
  };

}

#endif /* _OPENMBV_SPINEEXTRUSION_H_ */

