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

#include <openmbvcppinterface/doubleparam.h>
#include <openmbvcppinterface/object.h>
#include <cmath>

using namespace OpenMBV;
using namespace std;

DoubleParam::DoubleParam(int value_) : value((double)value_), paramStr("") {
}

DoubleParam::DoubleParam(double value_) : value(value_), paramStr("") {
}

DoubleParam::DoubleParam(std::string paramStr_) : value(0), paramStr(paramStr_) {
}

DoubleParam::DoubleParam(char *paramStr_) : value(0), paramStr(paramStr_) {
}

DoubleParam::DoubleParam(const char *paramStr_) : value(0), paramStr(paramStr_) {
}

DoubleParam::DoubleParam(std::string paramStr_, double value_) : value(value_), paramStr(paramStr_) {
  Object::addSimpleParameter(paramStr, value);
}

DoubleParam::DoubleParam(char *paramStr_, double value_) : value(value_), paramStr(paramStr_) {
  Object::addSimpleParameter(paramStr, value);
}

DoubleParam::DoubleParam(const char *paramStr_, double value_) : value(value_), paramStr(paramStr_) {
  Object::addSimpleParameter(paramStr, value);
}

DoubleParam::operator double() {
  if(paramStr=="")
    return value;
  else if(paramStr=="nan" || paramStr=="NaN" || paramStr=="NAN")
    return NAN;
  else
    return NAN;
}

std::ostream& OpenMBV::operator<<(std::ostream &os, const DoubleParam v) {
  if(v.paramStr=="")
    os<<v.value;
  else
    os<<v.paramStr;
  return os;
}
