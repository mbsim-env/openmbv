#ifndef _GROUP_H_
#define _GROUP_H_

#include <amviscppinterface/object.h>
#include <vector>
#include <H5Cpp.h>

namespace AMVis {

  class Group : public Object {
    friend class Body;
    protected:
      std::vector<Object*> object;
      bool separateFile;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
      void createHDF5File();
    public:
      Group(const std::string& name_);
      void addObject(Object* object);
      void setSeparateFile(bool sepFile) { separateFile=sepFile; }
      void initialize();
  };

}

#endif
