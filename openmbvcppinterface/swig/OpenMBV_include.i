#ifdef SWIGPYTHON
  // for python std::vector<double> wrapping work well out of the box
  %include "std_vector.i"
#endif

// include these headers to the wraper c++ source code (required to compile)
%{
#include <openmbvcppinterface/group.h>
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
#include <openmbvcppinterface/indexedfaceset.h>
#include <openmbvcppinterface/dynamicindexedfaceset.h>
%}
