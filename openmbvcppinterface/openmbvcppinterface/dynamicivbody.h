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

#ifndef _OPENMBV_DYNAICIVBODY_H_
#define _OPENMBV_DYNAICIVBODY_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>
#include <string>
#include <utility>

namespace OpenMBV {

  /** A object defined by a Open Inventor file or a VRML file with dynamic data from HDF5. */
  class DynamicIvBody : public Body {
    friend class ObjectFactory;
    public:
      /** The file of the iv file to read */
      void setIvFileName(std::string ivFileName_) { ivContent=""; ivFileName=std::move(ivFileName_); }
      std::string getIvFileName() { return ivFileName; }

      void setIvContent(std::string ivContent_) { ivFileName=""; ivContent=std::move(ivContent_); }
      const std::string& getIvContent() { return ivContent; }

      void setDataSize(size_t s) { dataSize = s; }
      size_t getDataSize() { return dataSize; }

      void setScalarData(bool s) { scalarData = s; }
      bool getScalarData() { return scalarData; }

      /** Remove all nodes of the name name from the iv file. */
      void addRemoveNodesByName(const std::string &name) { removeNodesByName.emplace_back(name); }

      std::vector<std::string> getRemoveNodesByName() { return removeNodesByName; }

      /** Remove all nodes of the type type from the iv file. */
      void addRemoveNodesByType(const std::string &type) { removeNodesByType.emplace_back(type); }

      std::vector<std::string> getRemoveNodesByType() { return removeNodesByType; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** Append a data vector the the h5 datsset */
      template<typename T>
      void append(const T& row) {
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override { return data?data->getRow(i):std::vector<double>(dataSize); }

      void setStateOffSet(const std::vector<double>& stateOff)
      {
        stateOffSet = stateOff;
      }

      std::vector<double> getStateOffSet() { return stateOffSet; }

    protected:
      DynamicIvBody();
      ~DynamicIvBody() override = default;
      std::string ivFileName;
      std::string ivContent;
      std::vector<std::string> removeNodesByName;
      std::vector<std::string> removeNodesByType;

      size_t dataSize;
      H5::VectorSerie<double>* data{nullptr};
      bool scalarData { false };

      /** optional offset for spine vector, may be used as inital position superposed by deflections or as static  */
      std::vector<double> stateOffSet;

      void createHDF5File() override;
      void openHDF5File() override;
  };

}

#endif
