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
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order given in
   * world frame (N is the number of spine points): time,
   * spine point 1 x, spine point 1 y, spine point 1 z, spine twist 1,
   * spine point 2 x, spine point 2 y, spine point 2 z, spine twist 2,
   * ...,
   * spine point N x, spine point N y, spine point N z, spine twist N */
  class SpineExtrusion : public DynamicColoredBody {
    public:
      /** constructor */
      SpineExtrusion();

      /** Retrun the class name */
      std::string getClassName() { return "SpineExtrusion"; }

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

      /** Set the 2D contour (cross-section) of the extrusion.
       * The contour (polygon) points must be in clockwise order. */
      void setContour(std::vector<PolygonPoint*> *contour_) { contour = contour_; }

      std::vector<PolygonPoint*>* getContour() { return contour; }

      /** Set the scale factor of the body. */
      void setScaleFactor(const ScalarParameter scale) {
        set(scaleFactor,scale);
      }

      double getScaleFactor() { return get(scaleFactor); }

      /** Append a data vector to the h5 datsset */
      void append(std::vector<double>& row) { 
        assert(data!=0); 
        data->append(row);
      }

      int getRows() { return data?data->getRows():-1; }
      std::vector<double> getRow(int i) { return data->getRow(i); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);

    protected:
      /** destructor */
      virtual ~SpineExtrusion();

      /** Number of spine points used for extrusion along a path. */
      int numberOfSpinePoints;

      /** Vector of local x-y points. */
      std::vector<PolygonPoint*> *contour;

      /** Each row comprises [time,spine world position,spine twist,...,spine world position,spine twist]. */
      H5::VectorSerie<double>* data;

      /** Scale factor of the body. */
      ScalarParameter scaleFactor;

      /** Write XML file for not time-dependent data. */
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
      
      /** Write H5 file for time-dependent data. */
      void createHDF5File();
      void openHDF5File();
  };

}

#endif /* _OPENMBV_SPINEEXTRUSION_H_ */

