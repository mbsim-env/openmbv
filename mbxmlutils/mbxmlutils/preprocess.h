#ifndef _MBXMLUTILS_PREPROCESS_H_
#define _MBXMLUTILS_PREPROCESS_H_

#include <fmatvec/atom.h>
#include <boost/locale.hpp>
#include <boost/bind.hpp>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutils/octeval.h>

namespace MBXMLUtils {

class Preprocess : public fmatvec::Atom {
  public:
    static void preprocess(boost::shared_ptr<MBXMLUtils::DOMParser> parser, OctEval &octEval, std::vector<boost::filesystem::path> &dependencies, xercesc::DOMElement *&e);
};

}

#endif
