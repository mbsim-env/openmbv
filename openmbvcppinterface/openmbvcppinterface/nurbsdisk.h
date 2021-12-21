/*
   OpenMBV - Open Multi Body Viewer.
   Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
   */

#ifndef _OPENMBV_NURBSDISK_H_
#define _OPENMBV_NURBSDISK_H_

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <hdf5serie/vectorserie.h>
#include <vector>
#include <cassert>
#include <stdexcept>

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
    friend class ObjectFactory;
    public:
      /** Draw reference frame of this object in the viewer if true (the default) */
      void setLocalFrame(bool f) { localFrameStr=(f)?"true":"false"; }

      bool getLocalFrame() { return localFrameStr=="true"?true:false; }

      /** Set the scale factor of the body. */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      double getScaleFactor() { return scaleFactor; }

      /** Set the number of points drawn between the nodes. */
      void setDrawDegree(int drawDegree_) {
        drawDegree=drawDegree_;
      }

      int getDrawDegree() { return drawDegree; }

      /** Set the inner and outer radius of the disk. */
      void setRadii(double Ri_, double Ro_) {
        Ri=Ri_;
        Ro=Ro_;
      }

      /** Set the inner radius of the disk. */
      void setRi(double Ri_) {
        Ri=Ri_;
      }

      /** Set the inner radius of the disk. */
      void setRo(double Ro_) {
        Ro=Ro_;
      }

      double getRi() { return Ri; }
      double getRo() { return Ro; }

      /** Set the azimuthal knot vector. 
       * These values should be set to the optimal circle values.
       */
      void setKnotVecAzimuthal(const std::vector<double> &KnotVecAzimuthal_) {
        KnotVecAzimuthal=KnotVecAzimuthal_;
      }

      std::vector<double> getKnotVecAzimuthal() { return KnotVecAzimuthal; }

      /** Set the radial knot vector. 
       * These value should be set to 1 each, resulting in a B-Spline curve.
       */
      void setKnotVecRadial(const std::vector<double> &KnotVecRadial_) {
        KnotVecRadial=KnotVecRadial_;
      }

      std::vector<double> getKnotVecRadial() { return KnotVecRadial; }

      /** Set the azimuthal number of finite elements used for drawing. */
      void setElementNumberAzimuthal(int ElementNumberAzimuthal_) {
        ElementNumberAzimuthal=ElementNumberAzimuthal_;
      }

      int getElementNumberAzimuthal() { return ElementNumberAzimuthal; }

      /** Set the radial number of finite elements used for drawing. */
      void setElementNumberRadial(int ElementNumberRadial_) {
        ElementNumberRadial=ElementNumberRadial_;
      }

      int getElementNumberRadial() { return ElementNumberRadial; }

      /** Set the degree of the interpolating splines in radial direction. */
      void setInterpolationDegreeRadial(int InterpolationDegreeRadial_) {
        InterpolationDegreeRadial=InterpolationDegreeRadial_;
      }

      int getInterpolationDegreeRadial() { return InterpolationDegreeRadial; }

      /** Set the degree of the interpolating splines in azimuthal direction. */
      void setInterpolationDegreeAzimuthal(int InterpolationDegreeAzimuthal_) {
        InterpolationDegreeAzimuthal=InterpolationDegreeAzimuthal_;
      }

      int getInterpolationDegreeAzimuthal() { return InterpolationDegreeAzimuthal; }
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
      template<typename T>
      void append(const T& row) { 
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object");
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override {
        int NodeDofs = (getElementNumberRadial() + 1) * (getElementNumberAzimuthal() + getInterpolationDegreeAzimuthal());
        return data?data->getRow(i):std::vector<double>(7+3*NodeDofs+3*getElementNumberAzimuthal()*drawDegree*2);
      }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      /** Write XML file for not time-dependent data. */
      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
    protected:
      NurbsDisk(); 
      ~NurbsDisk() override;

      /** Each row comprises [time,]. */
      H5::VectorSerie<double>* data{nullptr};

      /**
       * \brief String that contains, whether reference Frame should be drawn (="True") or not (="False")
       */
      std::string localFrameStr;

      /** Scale factor of the body. */
      double scaleFactor{1};

      /** Number of points drawn between the nodes. */
      int drawDegree{1};

      /** Inner and outer radius of disk */
      double Ri{0.};
      double Ro{0.};

      /** Number of finite elements in azimuthal and radial direction */
      int ElementNumberAzimuthal{0};
      int ElementNumberRadial{0};

      /** Degree of interpolating spline polynomials in radial and azimuthal direction */
      int InterpolationDegreeAzimuthal{8};
      int InterpolationDegreeRadial{3};

      /** Knot vector for azimuthal and radial direction */
      std::vector<double> KnotVecAzimuthal, KnotVecRadial;

      /** Normal of the disk in global coordinates */
      float *DiskNormal{nullptr};

      /** Point on the center of the disk used for visualisation*/
      float *DiskPoint{nullptr};

      /** Write H5 file for time-dependent data. */
      void createHDF5File() override;
      void openHDF5File() override;
  };

}

#endif /* _OPENMBV_NURBSDISK_H_ */

