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

#ifndef _OPENMBV_PATH_H_
#define _OPENMBV_PATH_H_

#include <openmbvcppinterface/body.h>
#include <hdf5serie/vectorserie.h>
#include <vector>
#include <stdexcept>

namespace OpenMBV {

  /** Draw a path of a reference point
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order given in
   * world frame: time, x, y, z */
  class Path : public Body {
    friend class ObjectFactory;
    protected:
      void createHDF5File() override;
      void openHDF5File() override;
      H5::VectorSerie<double>* data;
      std::vector<double> color;
      
      Path();
      ~Path() override;
    public:
      /** Append a data vector the to hf dataset */
      template<typename T>
      void append(const T& row) {
        if(data==nullptr) throw std::runtime_error("can not append data to an environment object");
        if(row.size()!=4) throw std::runtime_error("the dimension does not match");
        data->append(row);
      }

      int getRows() override { return data?data->getRows():0; }
      std::vector<double> getRow(int i) override { return data?data->getRow(i):std::vector<double>(4); }

      /** Set the color of the path (HSV values from 0 to 1). */
      void setColor(const std::vector<double>& hsv) {
        if(hsv.size()!=3) throw std::runtime_error("the dimension does not match");
        color=hsv;
      }

      std::vector<double> getColor() { return color; }

      /** Set the color of the path (HSV values from 0 to 1). */
      void setColor(double h, double s, double v) {
        std::vector<double> hsv;
        hsv.push_back(h);
        hsv.push_back(s);
        hsv.push_back(v);
        color=hsv;
      }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement* writeXMLFile(xercesc::DOMNode *parent) override;
  };

}

#endif
