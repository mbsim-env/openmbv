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

#include <vector>
#include <ostream>

#ifndef _OPENMBV_DOUBLEPARAM_H_
#define _OPENMBV_DOUBLEPARAM_H_

namespace OpenMBV {

  /** Stores a double or a string. */
  class DoubleParam {
    protected:
      double value;
      std::string paramStr;
    public:
      DoubleParam(int value_); // convenience: to prevent ambiguous conversion from int
      DoubleParam(double value_);
      DoubleParam(std::string paramStr_);
      DoubleParam(char *paramStr_);
      DoubleParam(const char *paramStr_);
      DoubleParam(std::string paramStr_, double value_);
      DoubleParam(char *paramStr_, double value_);
      DoubleParam(const char *paramStr_, double value_);
      operator double();
      friend std::ostream& operator<<(std::ostream &os, const DoubleParam v);
  };

  /** stream output of a double or string */
  std::ostream& operator<<(std::ostream &os, const DoubleParam v);

}

#endif
