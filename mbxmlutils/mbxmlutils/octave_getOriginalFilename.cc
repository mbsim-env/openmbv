/*
    getOriginalFilename octave DLD function

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
#include <boost/filesystem/path.hpp>

DEFUN_DLD(getOriginalFilename, args, nargout, "Get the (original) filename of the currently evaluated element") {
  if(args.length()!=0 || nargout!=1) {
    error("Must be called without arguments and one return value.");
    return {};
  }
  return octave_value(originalFilename.string());
}
