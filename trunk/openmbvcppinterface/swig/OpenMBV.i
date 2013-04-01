// module name
%module OpenMBV

// include special interfaces
%include "std_string.i"
%include "exception.i"

// for python std::vector<double> wrapping work well out of the box
#ifdef SWIGPYTHON
  %include "std_vector.i"
  %template(vector_double) std::vector<double>;
#endif

// for octave std::vector<double> needs special wrapping to octave vector/matrix
#ifdef SWIGOCTAVE
  %typemap(typecheck, precedence=200) std::vector<double>, const std::vector<double>& {
    $1=(*$input).is_matrix_type();
  }
  %typemap(in) std::vector<double>, const std::vector<double>& {
    Matrix m=$input.matrix_value();
    int size=m.length();
    static std::vector<double> localVec(size);
    for(int i=0; i<size; i++)
      localVec[i]=m.elem(i);
    $1=&localVec;
  }
  %typemap(out) std::vector<double> {
    RowVector rv($1.size());
    for(int i=0; i<$1.size(); i++)
      rv.fill($1[i], i, i);
    $result=rv;
  }
#endif

// wrap ScalarParameter to simple double value
%typemap(in) OpenMBV::ScalarParameter, const OpenMBV::ScalarParameter& {
  double value;
  if(SWIG_AsVal_double($input, &value)!=0)
    SWIG_exception(SWIG_TypeError, "Parameter of type double is expected. OpenMBV::ScalarParameter is not supported.");
  $1=value;
}

// generate interfaces for these files
%include <openmbvcppinterface/polygonpoint.h>
%include <openmbvcppinterface/object.h>
%include <openmbvcppinterface/group.h>
%include <openmbvcppinterface/body.h>
%include <openmbvcppinterface/dynamiccoloredbody.h>
%include <openmbvcppinterface/rigidbody.h>
%include <openmbvcppinterface/compoundrigidbody.h>
%include <openmbvcppinterface/spineextrusion.h>
%include <openmbvcppinterface/cube.h>
%include <openmbvcppinterface/nurbsdisk.h>
%include <openmbvcppinterface/rotation.h>
%include <openmbvcppinterface/arrow.h>
%include <openmbvcppinterface/ivbody.h>
%include <openmbvcppinterface/frustum.h>
%include <openmbvcppinterface/invisiblebody.h>
%include <openmbvcppinterface/frame.h>
%include <openmbvcppinterface/coilspring.h>
%include <openmbvcppinterface/sphere.h>
%include <openmbvcppinterface/extrusion.h>
%include <openmbvcppinterface/cuboid.h>
%include <openmbvcppinterface/grid.h>
%include <openmbvcppinterface/path.h>

// include these headers to the wraper c++ source code (required to compile)
%{
#include <openmbvcppinterface/polygonpoint.h>
#include <openmbvcppinterface/compoundrigidbody.h>
#include <openmbvcppinterface/spineextrusion.h>
#include <openmbvcppinterface/cube.h>
#include <openmbvcppinterface/nurbsdisk.h>
#include <openmbvcppinterface/rotation.h>
#include <openmbvcppinterface/arrow.h>
#include <openmbvcppinterface/ivbody.h>
#include <openmbvcppinterface/frustum.h>
#include <openmbvcppinterface/invisiblebody.h>
#include <openmbvcppinterface/frame.h>
#include <openmbvcppinterface/coilspring.h>
#include <openmbvcppinterface/sphere.h>
#include <openmbvcppinterface/extrusion.h>
#include <openmbvcppinterface/cuboid.h>
#include <openmbvcppinterface/grid.h>
#include <openmbvcppinterface/path.h>
%}
