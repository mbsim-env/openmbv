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
