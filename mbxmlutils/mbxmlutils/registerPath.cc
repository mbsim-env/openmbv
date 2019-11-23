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
