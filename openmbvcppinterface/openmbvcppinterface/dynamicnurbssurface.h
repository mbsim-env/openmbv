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

#ifndef _OPENMBV_DYNAMICNURBSSURFACE_H
#define _OPENMBV_DYNAMICNURBSSURFACE_H

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <vector>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** A dynamic nurbs surface */
  class DynamicNurbsSurface : public DynamicColoredBody {
    friend class ObjectFactory;
    protected:
      std::vector<std::vector<double> > cp;
      int numU{0}, numV{0};
      std::vector<double> uKnot, vKnot;
      H5::VectorSerie<double>* data;
      DynamicNurbsSurface() = default;
      ~DynamicNurbsSurface() override = default;
      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
      /** Write H5 file for time-dependent data. */
      void createHDF5File() override;
      void openHDF5File() override;
    public:
      /** Get control points
       */
      const std::vector<std::vector<double> >& getControlPoints() { return cp; }
      int getNumberOfUControlPoints() { return numU; }
      int getNumberOfVControlPoints() { return numV; }
      const std::vector<double>& getUKnotVector() { return uKnot; }
      const std::vector<double>& getVKnotVector() { return vKnot; }

      /** Set control points
       */
      void setControlPoints(const std::vector<std::vector<double> > &cp_) { cp = cp_; }
      void setNumberOfUControlPoints(int numU_) { numU = numU_; }
      void setNumberOfVControlPoints(int numV_) { numV = numV_; }
      void setUKnotVector(const std::vector<double>& uKnot_) { uKnot = uKnot_; }
      void setVKnotVector(const std::vector<double>& vKnot_) { vKnot = vKnot_; }

      /** Append a data vector to the h5 datsset */
      template<typename T>
      void append(const T& row) { 
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object"); 
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override { return data?data->getRow(i):std::vector<double>(1+4*numU*numV); }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;
  };

}

#endif
