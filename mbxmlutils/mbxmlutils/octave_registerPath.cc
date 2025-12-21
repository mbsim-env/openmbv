/*
    registerPath octave DLD function

    Copyright (C) Markus Friedrich

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

// octave used M_PI which is not longer defined in newer compilers
#define _USE_MATH_DEFINES
#include <cmath>

#include <octave/oct.h>
#include "mbxmlutils/eval_static.h"

DEFUN_DLD(registerPath, args, nargout, "Register a path as dependency of the current evaluator") {
  if(args.length()!=1 || nargout!=0) {
    error("Must be called with one argument and zero return values.");
    return octave_value_list();
  }
  if(!args(0).is_string()) {
    error("Argument must be of type string.");
    return octave_value_list();
  }
  mbxmlutilsStaticDependencies.push_back(args(0).string_value());
  return octave_value_list();
}
