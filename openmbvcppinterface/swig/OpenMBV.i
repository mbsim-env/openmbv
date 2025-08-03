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
%shared_ptr(OpenMBV::IndexedLineSet)
%shared_ptr(OpenMBV::IndexedFaceSet)
%shared_ptr(OpenMBV::FlexibleBody)
%shared_ptr(OpenMBV::DynamicIndexedLineSet)
%shared_ptr(OpenMBV::DynamicIndexedFaceSet)

// handle exceptions
%exception {
  try {
    $action
  }
  catch(const std::exception &e) {
    std::stringstream str;
    str<<e.what()<<std::endl;
    SWIG_exception(SWIG_RuntimeError, str.str().c_str());
  }
  catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

#ifdef SWIGPYTHON
  // for python std::vector<double> wrapping work well out of the box
  %template(VectorDouble) std::vector<double>;
  %template(VectorInt) std::vector<int>;
#endif

#ifdef SWIGOCTAVE
  // for octave std::vector<double> needs special wrapping to octave vector/matrix (cell array is the default)
  %typemap(typecheck, precedence=200) std::vector<double>, const std::vector<double>& {
    $1=(*$input).is_matrix_type();
  }
  %typemap(in) std::vector<double>, const std::vector<double>& {
    Matrix m=$input.matrix_value(); //MISSING: try do avoid copying all elements to m
    int size=m.rows();
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
  // handle vector<int>
  %typemap(typecheck, precedence=200) std::vector<int>, const std::vector<int>& {
    $1=(*$input).is_matrix_type();
  }
  %typemap(in) std::vector<int>, const std::vector<int>& {
    Matrix m=$input.matrix_value(); //MISSING: try do avoid copying all elements to m
    int size=m.rows();
    static std::vector<int> localVec;
    localVec.resize(size);
    for(int i=0; i<size; i++)//MISSING: try to avoid copying all element from m to localVec
      localVec[i]=m.elem(i);
    $1=&localVec;
  }
  %typemap(out) std::vector<int> {
    size_t n=$1.size();
    RowVector rv(n);
    for(int i=0; i<n; i++)
      rv.fill($1[i], i, i);
    $result=rv;
  }
  // handle vector<Index>
  %typemap(typecheck, precedence=200) std::vector<OpenMBV::Index>, const std::vector<OpenMBV::Index>& {
    $1=(*$input).is_matrix_type();
  }
  %typemap(in) std::vector<OpenMBV::Index>, const std::vector<OpenMBV::Index>& {
    Matrix m=$input.matrix_value();
    int size=m.rows();
    static std::vector<int> localVec;
    localVec.resize(size);
    for(int i=0; i<size; i++)
      localVec[i]=m.elem(i)-1;
    $1=&localVec;
  }
  %typemap(out) std::vector<OpenMBV::Index> {
    size_t n=$1.size();
    RowVector rv(n);
    for(int i=0; i<n; i++)
      rv.fill($1[i]+1, i, i);
    $result=rv;
  }
  // handle Index
  %typemap(typecheck, precedence=200) OpenMBV::Index, const OpenMBV::Index& {
    $1=(*$input).is_scalar_type();
  }
  %typemap(in) OpenMBV::Index, const OpenMBV::Index& {
    $1=static_cast<int>($input.scalar_value())-1;
  }
  %typemap(out) OpenMBV::Index {
    $result=$1+1;
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
    std::vector<double> vec$argnum(jenv->GetArrayLength($arg));
    $1=&vec$argnum;
    jenv->GetDoubleArrayRegion($arg, 0, vec$argnum.size(), vec$argnum.data());
  "
  %typemap(out) std::vector<double> {
    $result=jenv->NewDoubleArray($1.size());
    jenv->SetDoubleArrayRegion($result, 0, $1.size(), $1.data());
  }
  // on java we map std::vector<int> to int[] (this can than be mapped implcitly to a Matlab vector)
  %typemap(jni) std::vector<int>, const std::vector<int>& "jintArray"
  %typemap(jtype) std::vector<int>, const std::vector<int>& "int[]"
  %typemap(jstype) std::vector<int>, const std::vector<int>& "int[]"
  %typemap(javain) std::vector<int>, const std::vector<int>& "$javainput"
  %typemap(javaout) std::vector<int>, const std::vector<int>& { return $jnicall; }
  %typemap(in) std::vector<int>, const std::vector<int>& "
    std::vector<int> vec$argnum(jenv->GetArrayLength($arg));
    $1=&vec$argnum;
    jint v;
    for(size_t i=0; i<vec$argnum.size(); ++i) {
      jenv->GetIntArrayRegion($arg, i, 1, &v);
      vec$argnum[i]=v;
    }
  "
  %typemap(out) std::vector<int> {
    $result=jenv->NewIntArray($1.size());
    jint v;
    for(size_t i=0; i<$1.size(); ++i) {
      v=$1[i];
      jenv->SetIntArrayRegion($result, i, 1, &v);
    }
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
// swig scripting languages does not support implicit cast from boost::filesystem::path to std::string
// -> remove the path variant and add a string variant for setFileName
%ignore OpenMBV::Group::setFileName(const boost::filesystem::path &fn);
%extend OpenMBV::Group {
  void setFileName(const std::string &fn) {
    $self->setFileName(fn);
  }
}

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
%include <openmbvcppinterface/indexedlineset.h>
%include <openmbvcppinterface/indexedfaceset.h>
%include <openmbvcppinterface/flexiblebody.h>
%include <openmbvcppinterface/dynamicindexedlineset.h>
%include <openmbvcppinterface/dynamicindexedfaceset.h>

%extend OpenMBV::SpineExtrusion        { %template(append) append<std::vector<double> >; };
%extend OpenMBV::NurbsDisk             { %template(append) append<std::vector<double> >; };
%extend OpenMBV::Arrow                 { %template(append) append<std::vector<double> >; };
%extend OpenMBV::RigidBody             { %template(append) append<std::vector<double> >; };
%extend OpenMBV::CoilSpring            { %template(append) append<std::vector<double> >; };
%extend OpenMBV::Path                  { %template(append) append<std::vector<double> >; };
%extend OpenMBV::FlexibleBody          { %template(append) append<std::vector<double> >; };

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
%template(create_IndexedLineSet) OpenMBV::ObjectFactory::create<OpenMBV::IndexedLineSet>;
%template(create_IndexedFaceSet) OpenMBV::ObjectFactory::create<OpenMBV::IndexedFaceSet>;
%template(create_DynamicIndexedLineSet) OpenMBV::ObjectFactory::create<OpenMBV::DynamicIndexedLineSet>;
%template(create_DynamicIndexedFaceSet) OpenMBV::ObjectFactory::create<OpenMBV::DynamicIndexedFaceSet>;


// include these headers to the wraper c++ source code (required to compile)
%{
#include <openmbvcppinterface/objectfactory.h>
%}
