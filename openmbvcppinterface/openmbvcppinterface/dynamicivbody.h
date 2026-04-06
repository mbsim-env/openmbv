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

  /** A object defined by a Open Inventor file or a VRML file with dynamic data from HDF5.
   * In the IV file two special named nodes are accessible:
   * - "openmbv_body_outline_style": a node of type Group which can added to use the default outline style of OpenMBV.
   * - "openmbv_body_outline_switch": a node of type Switch. Its whichChild field is the current outline enable/disable flag of OpenMBV.
   */
  class DynamicIvBody : public Body {
    friend class ObjectFactory;
    public:
      /** The file of the iv file to read */
      void setIvFileName(std::string ivFileName_) { ivContent=""; ivFileName=std::move(ivFileName_); }
      std::string getIvFileName() { return ivFileName; }

      /** The content, as a string, of the iv data to read */
      void setIvContent(std::string ivContent_) { ivFileName=""; ivContent=std::move(ivContent_); }
      const std::string& getIvContent() { return ivContent; }

      /** The number of float data in the HDF5 file including the first data which must be the OpenMBV time.
       * The data can be accessed in the IV file as a node/field named "openmbv_dynamicivbody_data" or
       * "openmbv_dynamicivbody_data_0", "openmbv_dynamicivbody_data_1", see setScalarData. */
      void setDataSize(size_t s) { dataSize = s; }
      size_t getDataSize() { return dataSize; }

      /** The number of integer data in the HDF5 file.
       * The data can be accessed in the IV file as a node/field named "openmbv_dynamicivbody_dataInt" or
       * "openmbv_dynamicivbody_dataInt_0", "openmbv_dynamicivbody_dataInt_1", see setScalarData. */
      void setDataIntSize(size_t s) { dataIntSize = s; }
      size_t getDataIntSize() { return dataIntSize; }

      /** The number of string data in the HDF5 file.
       * The data can be accessed in the IV file as a node/field named "openmbv_dynamicivbody_dataStr" or
       * "openmbv_dynamicivbody_dataStr_0", "openmbv_dynamicivbody_dataStr_1", see setScalarData. */
      void setDataStrSize(size_t s) { dataStrSize = s; }
      size_t getDataStrSize() { return dataStrSize; }

      /** If true each data can be accessed using a separate node/single-value-field "..._<number>" see above.
       * If false each data can be accessed using a single node/multi-value-field "..." see above. */
      void setScalarData(bool s) { scalarData = s; }
      bool getScalarData() { return scalarData; }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** Append a data vector the the h5 datsset */
      template<typename T>
      void append(const T& row) {
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<Float> getRow(int i) override { return data?data->getRow(i):std::vector<Float>(dataSize); }

      /** Append a dataInt vector the the h5 datsset */
      template<typename T>
      void appendInt(const T& row) {
        dataInt->append(row);
      }

      std::vector<int> getRowInt(int i) { return dataInt?dataInt->getRow(i):std::vector<int>(dataIntSize); }

      /** Append a dataStr vector the the h5 datsset */
      template<typename T>
      void appendStr(const T& row) {
        dataStr->append(row);
      }

      std::vector<std::string> getRowStr(int i) { return dataStr?dataStr->getRow(i):std::vector<std::string>(dataStrSize); }

      template<typename T>
      void setStateOffSet(const std::vector<T>& stateOff) {
        stateOffSet.resize(stateOff.size());
        for(size_t i=0; i<stateOff.size(); ++i)
          stateOffSet[i] = stateOff[i];
      }
      std::vector<Float> getStateOffSet() { return stateOffSet; }

      void setStateIntOffSet(const std::vector<int>& stateOff) { stateIntOffSet = stateOff; }
      std::vector<int> getStateIntOffSet() { return stateIntOffSet; }

      void setStateStrOffSet(const std::vector<std::string>& stateOff) { stateStrOffSet = stateOff; }
      std::vector<std::string> getStateStrOffSet() { return stateStrOffSet; }

    protected:
      DynamicIvBody();
      ~DynamicIvBody() override = default;
      std::string ivFileName;
      std::string ivContent;

      size_t dataSize { 0 };
      size_t dataIntSize { 0 };
      size_t dataStrSize { 0 };
      H5::VectorSerie<Float>* data{nullptr};
      H5::VectorSerie<int>* dataInt{nullptr};
      H5::VectorSerie<std::string>* dataStr{nullptr};
      bool scalarData { false };

      /** optional offset for spine vector, may be used as inital position superposed by deflections or as static  */
      std::vector<Float> stateOffSet;
      std::vector<int> stateIntOffSet;
      std::vector<std::string> stateStrOffSet;

      void createHDF5File() override;
      void openHDF5File() override;
  };

}

#endif
