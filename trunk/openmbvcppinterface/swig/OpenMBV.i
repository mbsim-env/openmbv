// module name
%module OpenMBV

// include special interfaces
%include "std_string.i"
%include "exception.i"

// handle exceptions
%exception {
  try {
    $action
  }
  catch(const std::exception &e) {
    std::stringstream str;
    str<<"std::exception: "<<e.what();
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(const H5::Exception &e) {
    std::stringstream str;
    str<<"H5::Exception: "<<e.getCDetailMsg()<<std::endl<<
         "function: "<<e.getCFuncName();
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

// memory management
%delobject Object::destroy;
%delobject RigidBody::destroy;
%feature("unref") Object {
  if($this->getParent()==NULL)
    $this->destroy();
}

#ifdef SWIGPYTHON
  // for python std::vector<double> wrapping work well out of the box
  %include "std_vector.i"
  %template(VectorDouble) std::vector<double>;
#endif

#ifdef SWIGOCTAVE
  // for octave std::vector<double> needs special wrapping to octave vector/matrix (cell array is the default)
  %typemap(typecheck, precedence=200) std::vector<double>, const std::vector<double>& {
    $1=(*$input).is_matrix_type();
  }
  %typemap(in) std::vector<double>, const std::vector<double>& {
    Matrix m=$input.matrix_value(); //MISSING: try do avoid copying all elements to m
    int size=m.length();
    static std::vector<double> localVec;
    localVec.resize(size);
    for(int i=0; i<size; i++)//MISSING: try to avoid copying all element from m to localVec
      localVec[i]=m.elem(i);
    $1=&localVec;
  }
  %typemap(out) std::vector<double> {
    size_t n=$1.size();
    RowVector rv(n);
    for(int i=0; i<n; i++)
      rv.fill($1[i], i, i);
    $result=rv;
  }
#endif

#ifdef SWIGJAVA
  // on java we map std::vector<double> to double[] (this can than be mapped implcitly to a Matlab vector)
  %typemap(jni) std::vector<double>, const std::vector<double>& "jdoubleArray"
  %typemap(jtype) std::vector<double>, const std::vector<double>& "double[]"
  %typemap(jstype) std::vector<double>, const std::vector<double>& "double[]"
  %typemap(javain) std::vector<double>, const std::vector<double>& "$javainput"
  %typemap(javaout) std::vector<double> { return $jnicall; }
  %typemap(in) std::vector<double>, const std::vector<double>& "
    size_t size=jenv->GetArrayLength($arg);
    static std::vector<double> vec;
    vec.resize(size);
    jenv->GetDoubleArrayRegion($arg, 0, size, &vec[0]);//MISSING: try to avoid copying all elements to vec
    $1=&vec;
  "
  %typemap(out) std::vector<double> {
    $result=jenv->NewDoubleArray($1.size());
    jenv->SetDoubleArrayRegion($result, 0, $1.size(), &$1[0]);
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
#endif



#if defined SWIGJAVA
  // automatically load the native library
  %pragma(java) jniclasscode=%{
    private final static native void storeAndSetDLLSearchDirectory(String binDir);
    private final static native void restoreDLLSearchDirectory();
    static {
      try {
        final String fileName=OpenMBVJNI.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        final String binDir=fileName.substring(0, fileName.length()-new String("/openmbv.jar").length());
        System.load(binDir+"/libopenmbvjavaloadJNI.jni");
        storeAndSetDLLSearchDirectory(new java.io.File(binDir).toString());
        System.load(binDir+"/libopenmbvjava.jni");
        restoreDLLSearchDirectory();
      }
      catch(Throwable ex) {
        ex.printStackTrace();
        throw new RuntimeException("Unable to initialize. See above exception and stack trace.");
      }
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
