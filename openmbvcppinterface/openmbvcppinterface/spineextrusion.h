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
#include <cassert>
#include <hdf5serie/vectorserie.h>
#include <stdexcept>

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
    friend class ObjectFactory;
    public:
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
      void setContour(const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &contour_) { contour = contour_; }

      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > getContour() { return contour; }

      /** Set the scale factor of the body. */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      double getScaleFactor() { return scaleFactor; }

      /** Set the initial rotation of the body. */
      void setInitialRotation(const std::vector<double>& initRot) {
        if(initRot.size()!=3) throw std::runtime_error("the dimension does not match");
        initialRotation=initRot;
      }

      /** Set the initial rotation of the body. */
      void setInitialRotation(double a, double b, double g) {
        std::vector<double> initRot;
        initRot.push_back(a);
        initRot.push_back(b);
        initRot.push_back(g);
        initialRotation=initRot;
      }

      void setStateOffSet(const std::vector<double>& stateOff)
      {
        stateOffSet = stateOff;
      }

      std::vector<double> getStateOffSet( void ) { return stateOffSet; }

      /** Get the initial rotation of the body. */
      std::vector<double> getInitialRotation() { return initialRotation; }

      /** Append a data vector to the h5 datsset */
      template<typename T>
      void append(const T& row) { 
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object"); 
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override { return data?data->getRow(i):std::vector<double>(1+4*numberOfSpinePoints); }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      /** Write XML file for not time-dependent data. */
      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
    protected:
      SpineExtrusion();
      ~SpineExtrusion() override;

      /** Number of spine points used for extrusion along a path. */
      int numberOfSpinePoints{0};

      /** Vector of local x-y points. */
      std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > contour;

      /** Each row comprises [time,spine world position,spine twist,...,spine world position,spine twist]. */
      H5::VectorSerie<double>* data{nullptr};

      /** Scale factor of the body. */
      double scaleFactor{1};
      
      /** Intial rotation of the body. */
      std::vector<double> initialRotation;
      
      /** optional offset for spine vector, may be used as inital position superposed by deflections or as static  */
      std::vector<double> stateOffSet;
      
      /** Write H5 file for time-dependent data. */
      void createHDF5File() override;
      void openHDF5File() override;
  };

}

#endif /* _OPENMBV_SPINEEXTRUSION_H_ */

