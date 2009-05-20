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

#ifndef _OPENMBV_NURBSDISK_H_
#define _OPENMBV_NURBSDISK_H_

#include <openmbvcppinterface/body.h>
#include <vector>
#include <assert.h>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** 
   * \brief Class for all with a NURBS surface and a primitive closure
   * \author Kilian Grundl
   * \author Raphael Missel
   * \author Thorsten Schindler
   * \date 2009-05-20 initial commit (Grundl / Missel / Schindler)
   */
  class NurbsDisk : public Body {
    public:
      /** constructor */ 
      NurbsDisk(); // (REMARK[TS]: only standard constructor is allowed because of creation of OpenMBV bodies in ObjectFactory)

      /** destructor */
      virtual ~NurbsDisk();

      /** Set a static color for the body.
       * If this value is set, the color given to the append function
       * (as last element of the data row) is overwritten with this value.
       */
      void setStaticColor(const double col) {
        staticColor=col;
      }

      /** Set the minimal color value.
       * The color value of the body is linearly mapped between minimalColorValue
       * and maximalColorValue to blue(minimal) over cyan, green, yellow to red(maximal) [HSV-color model].
       */
      void setMinimalColorValue(const double min) {
        minimalColorValue=min;
      }

      /** Set the maximal color value.
       * See also minimalColorValue
       */
      void setMaximalColorValue(const double max) {
        maximalColorValue=max;
      }

      /** Set the scale factor of the body. */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      /** Set the inner and outer radius of the disk. */
      void setRadii(float Ri_, float Ro_) {
        Ri=Ri_;
        Ro=Ro_;
      }

      /** Set the inner and outer thickness of the disk. */
      void setThickness(float di_[3], float do_[3]) {
        // TODO 
        // REMARK[TS] really vector necessary?
      }

      /** Set the azimuthal knot vector. 
       * These values should be set to the optimal circle values.
       */
      void setKnotVecAzimuthal(float *KnotVecAzimuthal_) {
        KnotVecAzimuthal=KnotVecAzimuthal_;
      }

      /** Set the radial knot vector. 
       * These value should be set to 1 each, resulting in a B-Spline curve.
       */
      void setKnotVecRadial(float *KnotVecRadial_) {
        KnotVecRadial=KnotVecRadial_;
      }

      /** Set the azimuthal number of finite elements used for drawing. */
      void setElementNumberAzimuthal(int ElementNumberAzimuthal_) {
        // REMARK[TS]: necessary?
        ElementNumberAzimuthal=ElementNumberAzimuthal_;
      }

      /** Set the radial number of finite elements used for drawing. */
      void setElementNumberRadial(int ElementNumberRadial_) {
        // REMARK[TS]: necessary?
        ElementNumberRadial=ElementNumberRadial_;
      }

      /** Set the degree of the interpolating splines. */
      void setInterpolationDegree(int InterpolationDegree_) {
        InterpolationDegree=InterpolationDegree_;
      }

      /** Append a data vector to the h5 datsset */
      void append(std::vector<double>& row) { 
        assert(data!=0); 
        data->append(row);
      }

    protected:
      /** Each row comprises [time,spine world position,spine twist,...,spine world position,spine twist]. */
      H5::VectorSerie<double>* data;

      /** Static color value for all time. */
      double staticColor;

      /** Interpolation boundaries for colour mapping. */
      double minimalColorValue, maximalColorValue;

      /** Scale factor of the body. */
      double scaleFactor;

      /** Inner and outer radius of disk */
      float Ri, Ro;

      /** Knot vector for azimuthal and radial direction */
      float *KnotVecAzimuthal, *KnotVecRadial;

      /** Number of finite elements in azimuthal and radial direction */
      int ElementNumberAzimuthal, ElementNumberRadial;

      /** Degree of interpolating spline polynomials */
      int InterpolationDegree;

      /** Write XML file for not time-dependent data. */
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      
      /** Write H5 file for time-dependent data. */
      void createHDF5File();
  };

}

#endif /* _OPENMBV_NURBSDISK_H_ */

