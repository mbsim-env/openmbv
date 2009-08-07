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

#ifndef _OPENMBV_RIGIDBODY_H_
#define _OPENMBV_RIGIDBODY_H_

#include <openmbvcppinterface/dynamiccoloredbody.h>
#include <vector>
#include <assert.h>
#include <hdf5serie/vectorserie.h>
#include <cmath>

namespace OpenMBV {

  /** Abstract base class for all rigid bodies */
  class RigidBody : public DynamicColoredBody {
    friend class CompoundRigidBody;
    protected:
      std::vector<double> initialTranslation;
      std::vector<double> initialRotation;
      double scaleFactor;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
    public:
      /** Default constructor */
      RigidBody();

      ~RigidBody();

      /** Set initial translaton between the local frame of the body and the reference frame */
      void setInitialTranslation(const std::vector<double>& initTrans) {
        assert(initTrans.size()==3);
        initialTranslation=initTrans;
      }

      /** Set initial translaton between the local frame of the body and the reference frame */
      void setInitialTranslation(double x, double y, double z) {
        std::vector<double> initTrans;
        initTrans.push_back(x);
        initTrans.push_back(y);
        initTrans.push_back(z);
        initialTranslation=initTrans;
      }

      /** Set initial rotation between the local frame of the body and the reference frame.
       * Use cardan angles to represent the transformation matrix
       */
      void setInitialRotation(const std::vector<double>& initRot) {
        assert(initRot.size()==3);
        initialRotation=initRot;
      }

      /** Set initial rotation between the local frame of the body and the reference frame.
       * Use cardan angles to represent the transformation matrix
       */
      void setInitialRotation(double a, double b, double g) {
        std::vector<double> initRot;
        initRot.push_back(a);
        initRot.push_back(b);
        initRot.push_back(g);
        initialRotation=initRot;
      }

      /** Set the scale factor of the body */
      void setScaleFactor(const double scale) {
        scaleFactor=scale;
      }

      /** Append a data vector the the h5 datsset */
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==8);
        if(!std::isnan(staticColor)) row[7]=staticColor;
        data->append(row);
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif /* _OPENMBV_RIGIDBODY_H_ */

