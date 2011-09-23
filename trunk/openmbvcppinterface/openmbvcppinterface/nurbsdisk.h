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
   * \date 2010-08-09 adapt to new concept of Markus Friedrich (Schindler)
   */
  class NurbsDisk : public DynamicColoredBody {
    public:
      /** constructor */ 
      NurbsDisk(); 

      /** Retrun the class name */
      std::string getClassName() { return "NurbsDisk"; }

      /** Set the scale factor of the body. */
      void setScaleFactor(const ScalarParameter scale) {
        set(scaleFactor,scale);
      }

      double getScaleFactor() { return get(scaleFactor); }

      /** Set the number of points drawn between the nodes. */
      void setDrawDegree(const ScalarParameter drawDegree_) {
        set(drawDegree,drawDegree_);
      }

      double getDrawDegree() { return get(drawDegree); }

      /** Set the inner and outer radius of the disk. */
      void setRadii(ScalarParameter Ri_, ScalarParameter Ro_) {
        set(Ri,Ri_);
        set(Ro,Ro_);
      }

      double getRi() { return get(Ri); }
      double getRo() { return get(Ro); }

      /** Set the azimuthal knot vector. 
       * These values should be set to the optimal circle values.
       */
      void setKnotVecAzimuthal(const std::vector<double> &KnotVecAzimuthal_) {
        set(KnotVecAzimuthal,KnotVecAzimuthal_);
      }

      void setKnotVecAzimuthal(const VectorParameter &KnotVecAzimuthal_) {
        set(KnotVecAzimuthal,KnotVecAzimuthal_);
      }

      std::vector<double> getKnotVecAzimuthal() { return get(KnotVecAzimuthal); }

      /** Set the radial knot vector. 
       * These value should be set to 1 each, resulting in a B-Spline curve.
       */
      void setKnotVecRadial(const std::vector<double> &KnotVecRadial_) {
        set(KnotVecRadial,KnotVecRadial_);
      }

      void setKnotVecRadial(const VectorParameter &KnotVecRadial_) {
        set(KnotVecRadial,KnotVecRadial_);
      }

      std::vector<double> getKnotVecRadial() { return get(KnotVecRadial); }

      /** Set the azimuthal number of finite elements used for drawing. */
      void setElementNumberAzimuthal(ScalarParameter ElementNumberAzimuthal_) {
        set(ElementNumberAzimuthal,ElementNumberAzimuthal_);
      }

      int getElementNumberAzimuthal() { return int(get(ElementNumberAzimuthal)); }

      /** Set the radial number of finite elements used for drawing. */
      void setElementNumberRadial(ScalarParameter ElementNumberRadial_) {
        set(ElementNumberRadial,ElementNumberRadial_);
      }

      int getElementNumberRadial() { return int(get(ElementNumberRadial)); }

      /** Set the degree of the interpolating splines in radial direction. */
      void setInterpolationDegreeRadial(ScalarParameter InterpolationDegreeRadial_) {
        set(InterpolationDegreeRadial,InterpolationDegreeRadial_);
      }

      int getInterpolationDegreeRadial() { return int(get(InterpolationDegreeRadial)); }

      /** Set the degree of the interpolating splines in azimuthal direction. */
      void setInterpolationDegreeAzimuthal(ScalarParameter InterpolationDegreeAzimuthal_) {
        set(InterpolationDegreeAzimuthal,InterpolationDegreeAzimuthal_);
      }

      int getInterpolationDegreeAzimuthal() { return int(get(InterpolationDegreeAzimuthal)); }
      /** Set the global vector of the normal of the disk */
      void setDiskNormal(float *DiskNormal_) {
        DiskNormal=DiskNormal_;
      }

      float* getDiskNormal() { return DiskNormal; }

      /** Set the point in the center of the disk */
      void setDiskPoint(float *DiskPoint_) {
        DiskPoint=DiskPoint_;
      }

      float* getDiskPoint() { return DiskPoint; }

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
      virtual ~NurbsDisk();

      /** Each row comprises [time,]. */
      H5::VectorSerie<double>* data;

      /** Scale factor of the body. */
      ScalarParameter scaleFactor;

      /** Number of points drawn between the nodes. */
      ScalarParameter drawDegree;

      /** Inner and outer radius of disk */
      ScalarParameter Ri, Ro;

      /** Number of finite elements in azimuthal and radial direction */
      ScalarParameter ElementNumberAzimuthal, ElementNumberRadial;

      /** Degree of interpolating spline polynomials in radial and azimuthal direction */
      ScalarParameter InterpolationDegreeAzimuthal, InterpolationDegreeRadial;

      /** Knot vector for azimuthal and radial direction */
      VectorParameter KnotVecAzimuthal, KnotVecRadial;

      /** Normal of the disk in global coordinates */
      float *DiskNormal;

      /** Point on the center of the disk used for visualisation*/
      float *DiskPoint;

      /** Write XML file for not time-dependent data. */
      TiXmlElement* writeXMLFile(TiXmlNode *parent);

      /** Write H5 file for time-dependent data. */
      void createHDF5File();
      void openHDF5File();
  };

}

#endif /* _OPENMBV_NURBSDISK_H_ */

