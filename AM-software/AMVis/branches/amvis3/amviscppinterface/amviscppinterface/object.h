#ifndef _AMVIS_OBJECT_H_
#define _AMVIS_OBJECT_H_

#include <string>
#include <H5Cpp.h>

namespace AMVis {

  class Group;

  class Object {
    friend class Group;
    protected:
      std::string name;
      Group* parent;
      virtual void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="")=0;
      virtual void createHDF5File()=0;
      H5::Group *hdf5Group;
    public:
      Object();
      virtual ~Object();
      void setName(const std::string& name_) { name=name_; }
      std::string getFullName();
  };

}

#endif
