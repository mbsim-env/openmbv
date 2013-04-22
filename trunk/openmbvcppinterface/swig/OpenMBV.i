// module name
%module OpenMBV

// include special interfaces
%include "std_string.i"
%include "exception.i"

// for python and java std::vector<double> wrapping work well out of the box
#if defined SWIGPYTHON || defined SWIGJAVA
  %include "std_vector.i"
  %template(VectorDouble) std::vector<double>;
#endif

// for octave std::vector<double> needs special wrapping to octave vector/matrix (cell array is the default)
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

// SWIG typedefs for SimpleParameter template
namespace OpenMBV {
  template<class T>
  class SimpleParameter;
  typedef SimpleParameter<double> ScalarParameter;
  typedef SimpleParameter<std::vector<double> > VectorParameter;
  typedef SimpleParameter<std::vector<std::vector<double> > > MatrixParameter;
}

#if defined SWIGPYTHON || defined SWIGOCTAVE
  // allow for ScalarParameter on target side double and ScalarParameter values
  %typemap(in) OpenMBV::ScalarParameter, const OpenMBV::ScalarParameter& {
    double dblValue;
    OpenMBV::ScalarParameter *spPtr;
    if(SWIG_AsVal_double($input, &dblValue)==0)
      $1=dblValue;
    else if(SWIG_ConvertPtr($input, (void**)&spPtr, $descriptor(OpenMBV::ScalarParameter *), 0)==0)
      $1=*spPtr;
    else
      SWIG_exception(SWIG_TypeError, "Only double or OpenMBV::ScalarParameter is allowed as input.");
  }
#endif
#if defined SWIGJAVA
  // on java we map  ScalarParameter to double since implicit object convertion (by ctors) is not supported by java
  %typemap(jni) OpenMBV::ScalarParameter "double"
  %typemap(jtype) OpenMBV::ScalarParameter "double"
  %typemap(jstype) OpenMBV::ScalarParameter "double"
  %typemap(javain) OpenMBV::ScalarParameter "$javainput"
  %typemap(javaout) OpenMBV::ScalarParameter { return new ScalarParameter($jnicall).getValue(); }
  %typemap(in) OpenMBV::ScalarParameter { $1=OpenMBV::ScalarParameter($input); }

  // automatically load the native library
  %pragma(java) jniclasscode=%{
    static {
      final String fileName=OpenMBVJNI.class.getProtectionDomain().getCodeSource().getLocation().getPath();
      System.load(fileName.substring(0, fileName.length()-new String("openmbv.jar").length())+"libopenmbvjava.jni");
    }
  %}
#endif

// generate interfaces for these files
%include <openmbvcppinterface/simpleparameter.h>
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

// SWIG template instantiation
%rename(shiftLeft) operator<<;
%template(ScalarParameter) OpenMBV::SimpleParameter<double>;
%template(VectorParameter) OpenMBV::SimpleParameter<std::vector<double> >;
%template(MatrixParameter) OpenMBV::SimpleParameter<std::vector<std::vector<double> > >;

// include these headers to the wraper c++ source code (required to compile)
%{
#include <openmbvcppinterface/simpleparameter.h>
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
