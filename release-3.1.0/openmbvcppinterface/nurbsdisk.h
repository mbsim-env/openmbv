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

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <hdf5serie/vectorserie.h>
#include <vector>
#include <assert.h>

namespace OpenMBV {

  /** 
   * \brief Class for all bodies with a NURBS surface and a primitive closure
   * \author Kilian Grundl
   * \author Raphael Missel
   * \author Thorsten Schindler
   * \date 2009-05-20 initial commit (Grundl / Missel / Schindler)
   * \date 2009-08-16 visualisation / contour (Grundl / Missel / Schindler)
   */
  class NurbsDisk : public DynamicColoredBody {
    public:
      /** constructor */ 
      NurbsDisk(); 

      /** destructor */
      virtual ~NurbsDisk();

      /** Set the scale factor of the body. */
      void setScaleFactor(const DoubleParam scale) {
        scaleFactor=scale;
      }

      /** Set the number of points drawn between the nodes. */
      void setDrawDegree(const DoubleParam drawDegree_) {
        drawDegree=drawDegree_;
      }

      /** Set the inner and outer radius of the disk. */
      void setRadii(DoubleParam Ri_, DoubleParam Ro_) {
        Ri=Ri_;
        Ro=Ro_;
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
        ElementNumberAzimuthal=ElementNumberAzimuthal_;
      }

      /** Set the radial number of finite elements used for drawing. */
      void setElementNumberRadial(int ElementNumberRadial_) {
        ElementNumberRadial=ElementNumberRadial_;
      }

      /** Set the degree of the interpolating splines in radial direction. */
      void setInterpolationDegreeRadial(int InterpolationDegreeRadial_) {
        InterpolationDegreeRadial=InterpolationDegreeRadial_;
      }

      /** Set the degree of the interpolating splines in azimuthal direction. */
      void setInterpolationDegreeAzimuthal(int InterpolationDegreeAzimuthal_) {
        InterpolationDegreeAzimuthal=InterpolationDegreeAzimuthal_;
      }

      /** Set the global vector of the normal of the disk */
      void setDiskNormal(float *DiskNormal_) {
        DiskNormal=DiskNormal_;
      }

      /** Set the point in the center of the disk */
      void setDiskPoint(float *DiskPoint_) {
        DiskPoint=DiskPoint_;
      }

      /** Append a data vector to the h5 datsset */
      void append(std::vector<double>& row) { 
        assert(data!=0); 
        data->append(row);
      }

    protected:
      /** Each row comprises [time,]. */
      H5::VectorSerie<double>* data;

      /** Scale factor of the body. */
      DoubleParam scaleFactor;

      /** Number of points drawn between the nodes. */
      DoubleParam drawDegree;

      /** Inner and outer radius of disk */
      DoubleParam Ri, Ro;

      /** Knot vector for azimuthal and radial direction */
      float *KnotVecAzimuthal, *KnotVecRadial;

      /** Number of finite elements in azimuthal and radial direction */
      int ElementNumberAzimuthal, ElementNumberRadial;

      /** Degree of interpolating spline polynomials in radial and azimuthal direction */
      int InterpolationDegreeRadial, InterpolationDegreeAzimuthal;

      /** Normal of the disk in global coordinates */
      float *DiskNormal;

      /** Point on the center of the disk used for visualisation*/
      float *DiskPoint;

      /** Write XML file for not time-dependent data. */
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");

      /** Write H5 file for time-dependent data. */
      void createHDF5File();
  };

}

#endif /* _OPENMBV_NURBSDISK_H_ */

