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

#ifndef _OPENMBV_IVSCREENANNOTATION_H_
#define _OPENMBV_IVSCREENANNOTATION_H_

#include "body.h"
#include <hdf5serie/vectorserie.h>

namespace OpenMBV {

  /** A screen annotation defined by a Open Inventor file or a VRML file.
   *
   * This object is very special and behaves quite different compared to usual objects.
   * It not placed in the 3D scene and is not influenced by the camera or lights in the scene.
   * Hence, use e.g. a Material node with emissiveColor field to define the color of geometry.
   * It is placed in normalized screen coordinates. The x-axis is horizontal and the y-axis is vertical.
   * The bottom left coordinate is -1,-1 and the top right coordinate is 1,1.
   * The z-coordinate is irrelevant since z-buffering is disabled for these objects sicne its children are transparently render 
   * on top of the other. This means that a node defined later in the IV file will override previously nodes on the screen.
   * 
   * Note that the aspect ratio is only 1:1 if the screen is a square. However, it is possible to do an automatically scaling
   * to ensure that the aspect ratio is always 1:1, see setScale1To1At.
   *
   * The dynamic data of this object is defined by the user, see setColumnLabels. For each column a named node is created
   * which can be accessed from the IV file. The node name is the column label and the field "value" of this node holds the dynamic
   * data of the column.
   *
   * As e.g. RigidBody this object also provides drawing a path. The path point for each frame is defined by the origin of the node
   * named "OpenMBVIvScreenAnnotationPathOrigin" in the IV file. If no such node exists, no path is drawn.
   * If "OpenMBVIvScreenAnnotationPathOrigin" is given than also a node named "OpenMBVIvScreenAnnotationPathSep" of type Separator
   * must exist in the IV file. "OpenMBVIvScreenAnnotationPathSep" MUST NOT be influenced by any translation! However, its position
   * in the scene graph defines the order in with the path is drawn compared to the other nodes, see above.
   * The path is added as a child of this node.
   *
   * This object can also draw more than one path if the nodes are named "OpenMBVIvScreenAnnotationPathOrigin<nr>".
   * <nr> counts from 1 to N and stops if at the first node name which does not exist.
   * In this case the path is added as a child of the OpenMBVIvScreenAnnotationPathSep node, see above, or, if this node does not exist,
   * as a child of the OpenMBVIvScreenAnnotationPathSep<nr> node.
   *
   * All named nodes of name OpenMBVIvScreenAnnotation* are remove (during the ctor) from the
   * node/name mapping. */
  class IvScreenAnnotation : public Body {
    friend class ObjectFactory;
    public:
      void initializeUsingXML(xercesc::DOMElement *element) override;
      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
      void setScale1To1(bool scale1To1_);
      bool getScale1To1();

      /** Scales the screen to ensure that the screen coordinates have a aspect ratio of 1:1.
       * This scaling is done with a scale center at the specified point and the smaller axis is not scaled.
       * This means that the smaller axis still runs from -1 to 1.
       *
       * Use this function is you want to draw something with a aspect ratio of 1:1.
       * If you want the draw it e.g. in the top right corner set scale1To1Center_ to 1,1 and the coordinate of the
       * top right corner will still be 1,1 after the scaling to 1:1.
       */
      void setScale1To1At(const std::vector<double> &scale1To1Center_);

      std::vector<double> getScale1To1At();
      void setIvFileName(std::string ivFileName_) { ivContent=""; ivFileName=std::move(ivFileName_); }
      std::string getIvFileName() { return ivFileName; }
      void setIvContent(std::string ivContent_) { ivFileName=""; ivContent=std::move(ivContent_); }
      const std::string& getIvContent() { return ivContent; }
      void setColumnLabels(const std::vector<std::string> &columnLabels_);
      const std::vector<std::string>& getColumnLabels() const;

      void createHDF5File() override;
      void openHDF5File() override;

      template<typename T>
      void append(const T& row) {
        if(data==nullptr) throw std::runtime_error("IvScreenAnnotation: Cannot append data to an environment object");
        if(row.size()!=1+columnLabels.size()) throw std::runtime_error("IvScreenAnnotation: The dimension does not match (append: "+
                                                    std::to_string(row.size())+", columns: "+std::to_string(1+columnLabels.size())+")");
        data->append(&row[0], row.size());
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<Float> getRow(int i) override {
        if(!data)
          return std::vector<Float>(1+columnLabels.size());

        auto row = data->getRow(i);

        // handle legacy data (no "time" column as first column)
        if(columnLabels.size() == row.size()) {
          std::vector<Float> ret(1+row.size());
          ret[0] = 0; // use 0 for time column
          for(size_t i = 0; i < row.size(); ++i)
            ret[i+1] = row[i];
          return ret;
        }

        return row;
      }
    protected:
      IvScreenAnnotation();
      ~IvScreenAnnotation() override = default;
      bool scale1To1;
      std::vector<double> scale1To1Center;
      std::string ivFileName;
      std::string ivContent;
      std::vector<std::string> columnLabels;
      H5::VectorSerie<Float>* data{nullptr};
  };

}

#endif
