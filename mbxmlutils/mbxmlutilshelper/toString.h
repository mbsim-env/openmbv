/*
 * Author: Markus Friedrich
 *
 * This file is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <http://unlicense.org/>
 */

#ifndef _TOSTRING_H_
#define _TOSTRING_H_

#include <string>
#include <sstream>
#include <limits>

namespace MBXMLUtils {

template<typename T>
std::string toString(const T& value) {
  std::stringstream str;
  str.precision(std::numeric_limits<T>::digits10+1);
  str<<value;
  return str.str();
}

template<int N> using StringLiteral = char[N];
template<int N>
std::string toString(const StringLiteral<N>& value) {
  return value;
}

template<class T>
std::string toString(const std::vector<T>& value) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<T>::digits10+1);
  for(auto ele=value.begin(); ele!=value.end(); ++ele)
    oss<<(ele==value.begin()?"[":"; ")<< *ele;
  oss<<"]";
  return oss.str();
}

template<class T>
std::string toString(const std::vector<std::vector<T>>& value) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<T>::digits10+1);
  for(auto row=value.begin(); row!=value.end(); ++row)
    for(auto ele=row->begin(); ele!=row->end(); ++ele)
      oss<<(row==value.begin() && ele==row->begin()?"[":(ele==row->begin()?"; ":", "))<< *ele;
  oss<<"]";
  return oss.str();
}

}

#endif
