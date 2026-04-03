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

#ifndef _OPENMBV_DYNAMICATTRIBUTES_H_
#define _OPENMBV_DYNAMICATTRIBUTES_H_

#include <openmbvcppinterface/body.h>

#include <utility>
#include "hdf5serie/vectorserie.h"

namespace OpenMBV {

  /** \brief Control attributes of other objects dynamically.
   *
   * With this class you can define a list of objects, by its path,
   * for which a specific attribute should be controlled by the HDF5
   * data of this object.
   *
   * See the setter/adder methods which attributes of which objects
   * can be controlled.
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * single or double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order:
   *   - time
   *   - enable-attribute of 1st,2nd,... objectEnable (round(0.0)=disabled, else enabled)
   *   - draw-method-attribute of 1st,2nd,... bodyDrawMethod (round(0.0)=filled, round(1.0)=lines, round(2.0)=points)
   *   - transparency-attribute of 1st,2nd,... dynamicColoredBodyTransparency (0.0=opaque to 1.0=full-transparent)
   * If skip is true for a entry than this entry does not count in the HDF5 data, it uses the same data as
   * the first skip=false entry before.
   */
  class DynamicAttributes : public Body {
    friend class ObjectFactory;
    public:
#ifndef SWIG
      struct PathData {
        PathData(std::string path_, bool skip_) : path(std::move(path_)), skip(skip_) {}
        std::string path;
        bool skip;
      };
      using PathDataList = std::vector<PathData>;
#endif
    protected:
      DynamicAttributes();
      ~DynamicAttributes() override = default;

      PathDataList objectEnable;
      PathDataList bodyDrawMethod;
      PathDataList dynamicColoredBodyTransparency;

      void createHDF5File() override;
      void openHDF5File() override;
      H5::VectorSerie<Float>* data{nullptr};

      void updateDataSize();
      int dataSize;

    public:
      int getRows() override { return data?data->getRows():0; }
      std::vector<Float> getRow(int i) override { return data?data->getRow(i):std::vector<Float>(dataSize); }

      int getDataSize() { return dataSize; }

#ifndef SWIG
      /** Set the objects, by its path, for which the enable attribute should be controlled using HDF5 data. */
      void setObjectEnable(const PathDataList &p);
      const PathDataList getObjectEnable() const { return objectEnable; }
#endif
      /** Add a object, by its path, for which the enable attribute should be controlled using HDF5 data. */
      void addObjectEnable(const std::string &p, bool skip=false);

#ifndef SWIG
      /** Set the bodies, by its path, for which the draw-method attribute should be controlled using HDF5 data. */
      void setBodyDrawMethod(const PathDataList &p);
      const PathDataList getBodyDrawMethod() const { return bodyDrawMethod; }
#endif
      /** Add a body, by its path, for which the draw-method attribute should be controlled using HDF5 data. */
      void addBodyDrawMethod(const std::string &p, bool skip=false);

#ifndef SWIG
      /** Set the dynamic-colored-bodies, by its path, for which the transparency attribute should be controlled using HDF5 data. */
      void setDynamicColoredBodyTransparency(const PathDataList &p);
      const PathDataList getDynamicColoredBodyTransparency() const { return dynamicColoredBodyTransparency; }
#endif
      /** Add a dynamic-colored-body, by its path, for which the transparency attribute should be controlled using HDF5 data. */
      void addDynamicColoredBodyTransparency(const std::string &p, bool skip=false);

      void initializeUsingXML(xercesc::DOMElement *element) override;
      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;

      /** Append a data vector the the h5 datsset */
      template<typename T>
      void append(const T& row) {
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object");
        if(static_cast<int>(row.size())!=dataSize) throw std::runtime_error("the dimension does not match");
        data->append(row);
      }
  };

}

#endif
