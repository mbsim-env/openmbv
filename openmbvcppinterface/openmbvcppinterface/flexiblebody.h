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

#ifndef _OPENMBV_FLEXIBLEBODY_H
#define _OPENMBV_FLEXIBLEBODY_H

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <vector>
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** \brief Abstract base class for all flexible bodies
   */
  class FlexibleBody : public DynamicColoredBody {
    protected:
      int numvp{0};
      H5::VectorSerie<Float>* data;
      FlexibleBody() = default;
      ~FlexibleBody() override = default;
      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;
      void createHDF5File() override;
      void openHDF5File() override;
    public:
      /** Get number of vertex positions
       */
      int getNumberOfVertexPositions() const { return numvp; }

      /** Set number of vertex positions
       */
      void setNumberOfVertexPositions(int num) { numvp = num; }

      /** Append a data vector to the h5 datsset */
      template<typename T>
      void append(const T& row) { 
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object"); 
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<Float> getRow(int i) override { return data?data->getRow(i):std::vector<Float>(1+3*numvp); }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;
  };

}

#endif
