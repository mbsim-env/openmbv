// module name
%module OpenMBV

%include OpenMBV_include.i

// include special interfaces
%include "std_string.i"
%include "exception.i"

// std::shared_ptr
%include "std_shared_ptr.i"
%shared_ptr(OpenMBV::Object)
%shared_ptr(OpenMBV::Group)
%shared_ptr(OpenMBV::Body)
%shared_ptr(OpenMBV::DynamicColoredBody)
%shared_ptr(OpenMBV::RigidBody)
%shared_ptr(OpenMBV::CompoundRigidBody)
%shared_ptr(OpenMBV::SpineExtrusion)
%shared_ptr(OpenMBV::Cube)
%shared_ptr(OpenMBV::NurbsDisk)
%shared_ptr(OpenMBV::Rotation)
%shared_ptr(OpenMBV::Arrow)
%shared_ptr(OpenMBV::IvBody)
%shared_ptr(OpenMBV::Frustum)
%shared_ptr(OpenMBV::InvisibleBody)
%shared_ptr(OpenMBV::Frame)
%shared_ptr(OpenMBV::CoilSpring)
%shared_ptr(OpenMBV::Sphere)
%shared_ptr(OpenMBV::Extrusion)
%shared_ptr(OpenMBV::Cuboid)
%shared_ptr(OpenMBV::Grid)
%shared_ptr(OpenMBV::Path)

// handle exceptions
%exception {
  try {
    $action
  }
  catch(const std::exception &e) {
    std::stringstream str;
    str<<"Exception: "<<std::endl
       <<e.what()<<std::endl;
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

#ifdef SWIGPYTHON
  // for python std::vector<double> wrapping work well out of the box
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

%extend OpenMBV::SpineExtrusion { %template(append) append<std::vector<double> >; };
%extend OpenMBV::NurbsDisk      { %template(append) append<std::vector<double> >; };
%extend OpenMBV::Arrow          { %template(append) append<std::vector<double> >; };
%extend OpenMBV::RigidBody      { %template(append) append<std::vector<double> >; };
%extend OpenMBV::CoilSpring     { %template(append) append<std::vector<double> >; };
%extend OpenMBV::Path           { %template(append) append<std::vector<double> >; };

%include <openmbvcppinterface/objectfactory.h>
%template(create_Group) OpenMBV::ObjectFactory::create<OpenMBV::Group>;
%template(create_CompoundRigidBody) OpenMBV::ObjectFactory::create<OpenMBV::CompoundRigidBody>;
%template(create_SpineExtrusion) OpenMBV::ObjectFactory::create<OpenMBV::SpineExtrusion>;
%template(create_Cube) OpenMBV::ObjectFactory::create<OpenMBV::Cube>;
%template(create_NurbsDisk) OpenMBV::ObjectFactory::create<OpenMBV::NurbsDisk>;
%template(create_Rotation) OpenMBV::ObjectFactory::create<OpenMBV::Rotation>;
%template(create_Arrow) OpenMBV::ObjectFactory::create<OpenMBV::Arrow>;
%template(create_IvBody) OpenMBV::ObjectFactory::create<OpenMBV::IvBody>;
%template(create_Frustum) OpenMBV::ObjectFactory::create<OpenMBV::Frustum>;
%template(create_InvisibleBody) OpenMBV::ObjectFactory::create<OpenMBV::InvisibleBody>;
%template(create_Frame) OpenMBV::ObjectFactory::create<OpenMBV::Frame>;
%template(create_CoilSpring) OpenMBV::ObjectFactory::create<OpenMBV::CoilSpring>;
%template(create_Sphere) OpenMBV::ObjectFactory::create<OpenMBV::Sphere>;
%template(create_Extrusion) OpenMBV::ObjectFactory::create<OpenMBV::Extrusion>;
%template(create_Cuboid) OpenMBV::ObjectFactory::create<OpenMBV::Cuboid>;
%template(create_Grid) OpenMBV::ObjectFactory::create<OpenMBV::Grid>;
%template(create_Path) OpenMBV::ObjectFactory::create<OpenMBV::Path>;


// include these headers to the wraper c++ source code (required to compile)
%{
#include <openmbvcppinterface/objectfactory.h>
%}
