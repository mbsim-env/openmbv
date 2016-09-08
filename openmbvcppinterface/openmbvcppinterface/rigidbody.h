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
#include <stdexcept>

namespace OpenMBV {

  class CompoundRigidBody;

  /** \brief Abstract base class for all rigid bodies
   *
   * Each rigid body has a body fixed local coordinate system L
   * and a reference coordinate system R which has a fixed relative
   * position and rotation with respect to L (see figure).
   * And there is a inertial fixed world coordinate system W.
   *
   * \image html rigidbodycos.png "Coordinate Systems of Rigid Body"
   * \image latex rigidbodycos.eps "Coordinate Systems of Rigid Body" width=8cm
   *
   * The fixed translation from system R to system L is given by
   * the vector initialTranslation \f$[_R x_B, _R y_B, _R z_B]^T\f$
   * which coordinates are given in system R.
   *
   * The fixed rotation between the systems R and L is given by
   * the vector initialRotation \f$[\alpha_B, \beta_B, \gamma_B]^T\f$
   * which are the kardan angles of the transformation matrix 
   * \f[ A_{RL}= \textrm{cardan}(\alpha_B, \beta_B, \gamma_B) \f]
   * from system L to system R.
   *
   * The time dependend translation between the systems W and R is given
   * in the HDF5 dataset by the vector \f$ [_W x_P, _W y_P, _W z_P]^T=_W r_P \f$
   * which coordinates are given in system W.
   *
   * The time dependend rotation between the systems W and R is given
   * in the HDF5 dataset by the vector \f$ [\alpha_P, \beta_P, \gamma_P] \f$
   * which are the kardan angles of the transformation matrix
   * \f[ A_{WR}= \textrm{cardan}(\alpha_P, \beta_P, \gamma_P) \f]
   * from system R to system W.
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order: time,
   * \f$ _W x_P \f$, \f$ _W y_P \f$, \f$ _W z_P \f$,
   * \f$ \alpha_P \f$, \f$ \beta_P \f$, \f$ \gamma_P \f$,
   * color */
  class RigidBody : public DynamicColoredBody {
    friend class CompoundRigidBody;
    protected:
      std::string localFrameStr, referenceFrameStr, pathStr, draggerStr;
      std::vector<double> initialTranslation;
      std::vector<double> initialRotation;
      double scaleFactor;
      void createHDF5File();
      void openHDF5File();
      H5::VectorSerie<double>* data;
      std::weak_ptr<CompoundRigidBody> compound;

      RigidBody();
      ~RigidBody();
    public:
      /** Draw local frame of this object in the viewer if true (the default) */
      void setLocalFrame(bool f) { localFrameStr=(f==true)?"true":"false"; }

      bool getLocalFrame() { return localFrameStr=="true"?true:false; }

      /** Draw reference frame of this object in the viewer if true (the default) */
      void setReferenceFrame(bool f) { referenceFrameStr=(f==true)?"true":"false"; }

      bool getReferenceFrame() { return referenceFrameStr=="true"?true:false; }

      /** Draw path of this object in the viewer if true (the default) */
      void setPath(bool p) { pathStr=(p==true)?"true":"false"; }

      bool getPath() { return pathStr=="true"?true:false; }

      /** Draw dragger of this object in the viewer if true (default: false) */
      void setDragger(bool p) { draggerStr=(p==true)?"true":"false"; }

      bool getDragger() { return draggerStr=="true"?true:false; }

      /** Set initial translaton between the local frame of the body and the reference frame */
      void setInitialTranslation(const std::vector<double> &initTrans) {
        if(initTrans.size()!=3) std::runtime_error("the dimension does not match");
        initialTranslation=initTrans;
      }

      std::vector<double> getInitialTranslation() { return initialTranslation; }

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
        if(initRot.size()!=3) throw std::runtime_error("the dimension does not match");
        initialRotation=initRot;
      }

      std::vector<double> getInitialRotation() { return initialRotation; }

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
      void setScaleFactor(double scale) {
        scaleFactor=scale;
      }

      double getScaleFactor() { return scaleFactor; }

      /** Append a data vector the the h5 datsset */
      template<typename T>
      void append(const T& row) {
        if(data==0) throw std::runtime_error("can not append data to an environment object");
        if(row.size()!=8) throw std::runtime_error("the dimension does not match");
        if(!std::isnan(dynamicColor))
        {
          std::vector<double> tmprow(8);
          std::copy(&row[0], &row[8], tmprow.begin());
          tmprow[7]=dynamicColor;
          data->append(tmprow);
        }
        else
          data->append(row);
      }

      int getRows() { return data?data->getRows():-1; }
      std::vector<double> getRow(int i) { return data?data->getRow(i):std::vector<double>(8); }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(xercesc::DOMElement *element);

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent);

      virtual std::string getFullName(bool includingFileName=false, bool stopAtSeparateFile=false);

      std::shared_ptr<Group> getSeparateGroup();
      std::shared_ptr<Group> getTopLevelGroup();
  };

}

#endif /* _OPENMBV_RIGIDBODY_H_ */

