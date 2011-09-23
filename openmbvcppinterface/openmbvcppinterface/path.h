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
#include <openmbvcppinterface/simpleparameter.h>
#include <hdf5serie/vectorserie.h>
#include <vector>

namespace OpenMBV {

  /** Draw a path of a reference point
   *
   * HDF5-Dataset: The HDF5 dataset of this object is a 2D array of
   * double precision values. Each row represents one dataset in time.
   * A row consists of the following columns in order given in
   * world frame: time, x, y, z */
  class Path : public Body {
    protected:
      TiXmlElement* writeXMLFile(TiXmlNode *parent);
      void createHDF5File();
      void openHDF5File();
      H5::VectorSerie<double>* data;
      VectorParameter color;
      
      /** Destructor */
      virtual ~Path();
    public:
      /** Default constructor */
      Path();

      /** Retrun the class name */
      std::string getClassName() { return "Path"; }

      /** Append a data vector the to hf dataset */
      void append(std::vector<double>& row) {
        assert(data!=0 && row.size()==4);
        data->append(row);
      }

      int getRows() { return data?data->getRows():-1; }
      std::vector<double> getRow(int i) { return data->getRow(i); }

      /** Set the color of the paht.
       * Use a vector with tree double representing reg, green and blue as paremter.
       * red, green and blue runs form 0 to 1
       */
      void setColor(const VectorParameter& color_) {
        assert(color_.getParamStr()!="" || color_.getValue().size()==3);
        set(color,color_);
      }

      /** Set the color of the paht.
       * Use a vector with tree double representing reg, green and blue as paremter.
       * red, green and blue runs form 0 to 1
       */
      void setColor(const std::vector<double>& color_) {
        assert(color_.size()==3);
        set(color,color_);
      }

      std::vector<double> getColor() { return get(color); }

      /** Set the color of the paht.
       * red, green and blue runs form 0 to 1
       */
      void setColor(double red, double green, double blue) {
        std::vector<double> c;
        c.push_back(red);
        c.push_back(green);
        c.push_back(blue);
        color=c;
      }

      /** Initializes the time invariant part of the object using a XML node */
      virtual void initializeUsingXML(TiXmlElement *element);
  };

}

#endif
