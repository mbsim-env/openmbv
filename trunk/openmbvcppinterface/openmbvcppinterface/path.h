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

namespace OpenMBV {

  /** Draw a path of a reference point */
  class Path : public Body {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
      H5::VectorSerie<double>* data;
      std::vector<double> color;
    public:
      /** Default constructor */
      Path();

      /** Append a data vector the to hf dataset */
      void append(const std::vector<double>& row) {
        assert(data!=0 && row.size()==4);
        data->append(row);
      }

      /** Set the color of the paht.
       * Use a vector with tree double representing reg, green and blue as paremter.
       * red, green and blue runs form 0 to 1
       */
      void setColor(const std::vector<double>& color_) {
        assert(color_.size()==3);
        color=color_;
      }

      /** Set the color of the paht.
       * red, green and blue runs form 0 to 1
       */
      void setColor(double red, double green, double blue) {
        color.clear();
        color.push_back(red);
        color.push_back(green);
        color.push_back(blue);
      }
  };

}

#endif
