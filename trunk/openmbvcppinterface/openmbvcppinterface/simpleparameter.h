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
#include <openmbvcppinterface/group.h>

#ifndef _OPENMBV_SIMPLEPARAMETER_H_
#define _OPENMBV_SIMPLEPARAMETER_H_

namespace OpenMBV {

  typedef SimpleParameter<double> ScalarParameter;
  typedef SimpleParameter<std::vector<double> > VectorParameter;
  typedef SimpleParameter<std::vector<std::vector<double> > > MatrixParameter;

  template<class T>
  class SimpleParameter {
    friend class Object;
    protected:
      T value;
      std::string paramStr;
      bool addParameter;
      SimpleParameter(const std::string& paramStr_) : value(T()), paramStr(paramStr_), addParameter(false) {}
    public:
      SimpleParameter(const T& value_) : value(value_), paramStr(""), addParameter(true) {}
      SimpleParameter(const std::string& paramStr_, const T& value_) : value(value_), paramStr(paramStr_), addParameter(true) {}
      friend std::ostream& operator<<(std::ostream &os, const SimpleParameter<T> p) {
        if(p.paramStr=="")
          os<<p.value;
        else
          os<<p.paramStr;
        return os;
      };
      std::string getParamStr() const { return paramStr; }
      T getValue() const { return value; }
      bool getAddParameter() { return addParameter; }
  };

  template<class T>
  std::ostream& operator<<(std::ostream &os, const std::vector<T> p) {
    os<<"[";
    for(size_t i=0; i<p.size()-1; i++)
      os<<p[i]<<";";
    os<<p[p.size()-1]<<"]";
    return os;
  }

}

#endif
