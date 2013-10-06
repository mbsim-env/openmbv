#include "mbxmlutils/octeval.h"
#include "mbxmlutilstinyxml/tinyxml.h"
#include <stdexcept>
#include <boost/filesystem.hpp>
#include "mbxmlutilstinyxml/tinynamespace.h"
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <mbxmlutilstinyxml/utils.h>
#include <octave/parse.h>
#include <octave/octave.h>
#include <boost/scope_exit.hpp>
#include <casadi/symbolic/sx/sx_tools.hpp>

//MFMF: should also compile if casadi is not present; check throw statements

using namespace std;

namespace MBXMLUtils {

template<int T>
class Block {
  public:
    Block(ostream &str_) : str(str_) {
      if(disableCount==0)
        orgcxxx=str.rdbuf(0);
      disableCount++;
    }
    ~Block() {
      disableCount--;
      if(disableCount==0)
        str.rdbuf(orgcxxx);
    }
  private:
    ostream &str;
    static streambuf *orgcxxx;
    static int disableCount;
};
template<int T> streambuf *Block<T>::orgcxxx;
template<int T> int Block<T>::disableCount=0;
#define BLOCK_STDOUT Block<1> mbxmlutils_dummy_blockstdout(std::cout);
#define BLOCK_STDERR Block<2> mbxmlutils_dummy_blockstderr(std::cerr);

class PreserveCurrentDir {
  public:
    PreserveCurrentDir() {
      dir=boost::filesystem::current_path();
    }
    ~PreserveCurrentDir() {
      boost::filesystem::current_path(dir);
    }
  private:
    boost::filesystem::path dir;
};

template<>
string OctEval::cast<string>(const octave_value &value) {
  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(getType(value)==ScalarType) {
    ret<<value.double_value();
  }
  else if(getType(value)==VectorType || getType(value)==MatrixType) {
    Matrix m=value.matrix_value();
    ret<<"[";
    for(int i=0; i<m.rows(); i++) {
      for(int j=0; j<m.cols(); j++)
        ret<<m(j*m.rows()+i)<<(j<m.cols()-1?",":"");
      ret<<(i<m.rows()-1?" ; ":"]");
    }
    if(m.rows()==0)
      ret<<"]";
  }
  else if(getType(value)==StringType) {
    ret<<"'"<<value.string_value()<<"'";
  }
  else // if not scalar, matrix or string => error
    throw runtime_error("Can not convert this octave variable to a string.");

  return ret.str();
}

template<typename T>
T *OctEval::getSwigObjectPtr(const octave_value &value) {
  // get the casadi pointer: octave returns a 64bit integer which represents the pointer
  static octave_function *swig_this=symbol_table::find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  octave_value swigThis=feval(swig_this, value, 1)(0);
  if(error_state!=0) { error_state=0; throw runtime_error("Internal error: unable to get the swig pointer."); }
  void *swigPtr=NULL;
  swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  // convert the void pointer to the correct casadi type
  return static_cast<T*>(swigPtr);
}

namespace {
  void noDelete(CasADi::SXMatrix *p) {}
}

template<>
OctEval::SXMatrixRef OctEval::cast<OctEval::SXMatrixRef>(const octave_value &value) {
  ValueType type=getType(value);
  if(type!=ScalarType && type!=VectorType && type!=MatrixType && type!=SXMatrixType)
    throw runtime_error("Unknown type: must be a scalar, vector, matrix or CasADi SXMatrix type.");

  // for scalar, vector and matrix create a new equalent SXMatrix object and return a reference counted reference to it.
  if(type==ScalarType)
    return SXMatrixRef(new CasADi::SXMatrix(1,1,value.double_value()));
  if(type==VectorType || type==MatrixType) {
    Matrix m=value.matrix_value();
    SXMatrixRef mat(new CasADi::SXMatrix(m.rows(), m.cols()));
    for(int i=0; i<m.rows(); i++)
      for(int j=0; j<m.cols(); j++)
        (*mat)[i][j]=m(j*m.rows()+i);
    return mat;
  }

  // return a referent to the SXMatrix object stored in octave but to not delete it since octave owns the object
  return SXMatrixRef(getSwigObjectPtr<CasADi::SXMatrix>(value), &noDelete);
}

template<>
CasADi::SXFunction OctEval::cast<CasADi::SXFunction>(const octave_value &value) {
  ValueType type=getType(value);
  if(type!=SXFunctionType)
    throw runtime_error("Unknown type: must be a CasADi SXFunction type.");

  return *getSwigObjectPtr<CasADi::SXFunction>(value);
}

octave_value OctEval::createSXMatrix(const string &name, int dim1, int dim2) {
  static list<octave_value_list> idx;
  static octave_value *name_, *dim1_, *dim2_;
  if(idx.empty()) {
    idx.push_back(octave_value_list("ssym"));
    octave_value_list arg;
    arg.append("dummy");
    arg.append(0);
    arg.append(0);
    idx.push_back(arg);
    name_=&((*(--idx.end()))(0));
    dim1_=&((*(--idx.end()))(1));
    dim2_=&((*(--idx.end()))(2));
  }
  *name_=name;
  *dim1_=dim1;
  *dim2_=dim2;
  return casadiOctValue.subsref(".(", idx);
}

octave_value OctEval::createSXFunction(const Cell &inputs, const octave_value &output) {
  static list<octave_value_list> idx;
  static octave_value *inputs_, *outputs_;
  if(idx.empty()) {
    idx.push_back(octave_value_list("SXFunction"));
    octave_value_list arg;
    arg.append(octave_value(Cell()));
    arg.append(octave_value(Cell(1,1)));
    idx.push_back(arg);
    inputs_ =&((*(--idx.end()))(0));
    outputs_=&((*(--idx.end()))(1));
  }
  *inputs_ =inputs;
  *outputs_=Cell(output);
  return casadiOctValue.subsref(".(", idx);
}

OctEvalException::OctEvalException(const std::string &msg_, const TiXmlElement *e, const std::string &attrName) {
  string pre=": ";
  msg=TiXml_location_vec(e, "", pre+msg_);
}

void OctEvalException::print() const {
  for(vector<string>::const_iterator i=msg.begin(); i!=msg.end(); i++)
    cerr<<*i<<endl;
}

int OctEval::initCount=0;
std::map<std::string, std::string> OctEval::units;

OctEval::OctEval() {
  if(initCount==0) {

    string XMLDIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/xml"; // use rel path if build configuration dose not work
  
    static vector<char*> octave_argv;
    octave_argv.resize(6);
    octave_argv[0]=const_cast<char*>("embedded");
    octave_argv[1]=const_cast<char*>("--no-history");
    octave_argv[2]=const_cast<char*>("--no-init-file");
    octave_argv[3]=const_cast<char*>("--no-line-editing");
    octave_argv[4]=const_cast<char*>("--no-window-system");
    octave_argv[5]=const_cast<char*>("--silent");
    octave_main(6, &octave_argv[0], 1);
  
    octave_value_list warnArg;
    warnArg.append("error");
    warnArg.append("Octave:divide-by-zero");
    feval("warning", warnArg);
    if(error_state!=0) { error_state=0; throw runtime_error("Internal error: unable to disable warnings."); }
  
    feval("addpath", octave_value_list(octave_value(MBXMLUtils::getInstallPath()+"/share/mbxmlutils/octave")));
    if(error_state!=0) { error_state=0; throw runtime_error("Internal error: cannot add octave search path."); }

    feval("addpath", octave_value_list(octave_value(MBXMLUtils::getInstallPath()+"/bin")));
    if(error_state!=0) { error_state=0; throw string("Internal error: cannot add casadi octave search path."); }

    {
      BLOCK_STDERR;
      feval("casadi");
      if(error_state!=0) { error_state=0; throw string("Internal error: unable to initialize casadi."); }
      casadiOctValue=symbol_table::varref("casadi");
    }

    // get units
    cout<<"Build unit list for measurements."<<endl;
    boost::shared_ptr<TiXmlDocument> mmdoc(new TiXmlDocument);
    mmdoc->LoadFile(XMLDIR+"/measurement.xml"); TiXml_PostLoadFile(mmdoc.get());
    TiXmlElement *ele, *el2;
    for(ele=mmdoc->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement())
      for(el2=ele->FirstChildElement(); el2!=0; el2=el2->NextSiblingElement()) {
        if(units.find(el2->Attribute("name"))!=units.end())
          throw runtime_error(string("Internal error: Unit name ")+el2->Attribute("name")+" is defined more than once.");
        units[el2->Attribute("name")]=el2->GetText();
      }
  }
  initCount++;
};

OctEval::~OctEval() {
  initCount--;
  if(initCount==0) {
    //Workaround: eval a VALID dummy statement before leaving "main" to prevent a crash in post main
    int dummy;
    eval_string("1+1", true, dummy, 0); // eval as statement list
  }
}

void OctEval::addParam(const std::string &paramName, const octave_value& value) {
  currentParam[paramName]=value;
}

void OctEval::addParamSet(const TiXmlElement *e) {
  // outer loop to resolve recursive parameters
  list<const TiXmlElement*> c;
  for(const TiXmlElement *ee=e->FirstChildElement(); ee!=NULL; ee=ee->NextSiblingElement())
    c.push_back(ee);
  size_t length=c.size();
  for(size_t outerLoop=0; outerLoop<length; outerLoop++) {
    // evaluate parameter
    list<const TiXmlElement*>::iterator ee=c.begin();
    while(ee!=c.end()) {
      int err=0;
      octave_value ret;
      try { 
        BLOCK_STDERR;
        ret=eval(*ee);
      }
      catch(const OctEvalException &ex) {
        err=1;
      }
      if(err==0) { // if no error
        addParam((*ee)->Attribute("name"), ret); // add param to list
        list<const TiXmlElement*>::iterator eee=ee; eee++;
        c.erase(ee);
        ee=eee;
      }
      else
        ee++;
    }
  }
  if(c.size()>0) { // if parameters are left => error
    cerr<<"Error in one of the following parameters or infinit loop in this parameters:\n";
    for(list<const TiXmlElement*>::iterator ee=c.begin(); ee!=c.end(); ee++) {
      try {
        eval(*ee);
      }
      catch(const OctEvalException &ex) {
        ex.print();
      }
    }
    throw runtime_error("Error processing parameters. See above.");
  }
}

void OctEval::pushParams() {
  paramStack.push(currentParam);
}

void OctEval::popParams() {
  currentParam=paramStack.top();
  paramStack.pop();
}

octave_value OctEval::stringToOctValue(const std::string &str, const TiXmlElement *e) const {
  // clear octave
  symbol_table::clear_variables();
  // restore current parameters
  for(map<string, octave_value>::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    symbol_table::varref(i->first)=i->second;

  {
    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    const TiXmlElement *base=TiXml_GetElementWithXmlBase(e, 0);
    if(base) // set working dir to path of current file, so that octave works with correct relative paths
      boost::filesystem::current_path(fixPath(base->Attribute("xml:base"), "."));

    int dummy;
    try{
      BLOCK_STDOUT;
      eval_string(str, true, dummy, 0); // eval as statement list
    }
    catch(...) {
      error_state=1;
    }
  }
  if(error_state!=0) { // if error => wrong code => throw error
    error_state=0;
    throw OctEvalException("Unable to evaluate expression: "+str, e);
  }
  // generate a strNoSpace from str by removing leading/trailing spaces as well as trailing ';'.
  string strNoSpace=str;
  while(strNoSpace.size()>0 && strNoSpace[0]==' ')
    strNoSpace=strNoSpace.substr(1);
  while(strNoSpace.size()>0 && (strNoSpace[strNoSpace.size()-1]==' ' || strNoSpace[strNoSpace.size()-1]==';'))
    strNoSpace=strNoSpace.substr(0, strNoSpace.size()-1);
  if(!symbol_table::is_variable("ret") && !symbol_table::is_variable("ans") && !symbol_table::is_variable(strNoSpace)) {
    throw OctEvalException("'ret' variable not defined in multi statement octave expression or incorrect single statement: "+
      str, e);
  }
  octave_value ret;
  if(symbol_table::is_variable(strNoSpace))
    ret=symbol_table::varref(strNoSpace);
  else if(!symbol_table::is_variable("ret"))
    ret=symbol_table::varref("ans");
  else
    ret=symbol_table::varref("ret");

  return ret;
}

octave_value OctEval::eval(const TiXmlElement *e, const string &attrName, bool fullEval) {
  // handle attribute attrName
  if(!attrName.empty()) {
    // evaluate attribute fully
    if(fullEval) {
      octave_value ret=stringToOctValue(e->Attribute(attrName.c_str()), e);
      return ret;
    }

    // evaluate attribute only partially
    else {
      string s=attrName;
      size_t i;
      while((i=s.find('{'))!=string::npos) {
        size_t j=i;
        do {
          j=s.find('}', j+1);
          if(j==string::npos) throw runtime_error("no matching } found in attriubte.");
        }
        while(s[j-1]!='\\'); // skip } which is quoted with backslash
        string evalStr=s.substr(i+1,j-i-1);
        // remove the backlash quote from { and }
        size_t k=0;
        while((k=evalStr.find('{', k))!=string::npos) {
          if(k==0 || evalStr[k-1]!='\\') throw runtime_error("{ must be quoted with a backslash inside {...}.");
          evalStr=evalStr.substr(0, k-1)+evalStr.substr(k);
        }
        k=0;
        while((k=evalStr.find('}', k))!=string::npos) {
          if(k==0 || evalStr[k-1]!='\\') throw runtime_error("} must be quoted with a backslash inside {...}.");
          evalStr=evalStr.substr(0, k-1)+evalStr.substr(k);
        }
        
        octave_value ret=stringToOctValue(evalStr, e);
        s=s.substr(0,i)+cast<string>(ret)+s.substr(j+1);
      }
      return s;
    }
  }

  // handle element e
  else {
    const TiXmlElement *ec;
    bool function=e->Attribute("arg1name")!=NULL;

    // for functions add the function arguments as parameters
    if(function) pushParams();
    OctEval *_this=this;
    BOOST_SCOPE_EXIT((&function)(&_this)) {
      if(function) _this->popParams();
    } BOOST_SCOPE_EXIT_END
    Cell inputs;
    if(function) {
      addParam("casadi", casadiOctValue);
      // loop over all attributes and search for arg1name, arg2name attributes
      for(const TiXmlAttribute *a=e->FirstAttribute(); a!=NULL; a=a->Next()) {
        string value=a->Name();
        if(value.substr(0,3)=="arg" && value.substr(value.length()-4,4)=="name") {
          int nr=atoi(value.substr(3, value.length()-4-3).c_str());
          int dim=1;
          stringstream str;
          str<<"arg"<<nr<<"dim";
          if(e->Attribute(str.str())) dim=atoi(e->Attribute(str.str().c_str()));

          octave_value octArg=createSXMatrix(a->Value(), dim, 1);
          addParam(a->Value(), octArg);
          inputs.resize1(max(nr, inputs.dim2()), false); // fill new elements with bool "false" value (detect these later)
          inputs.elem(nr-1)=octArg;
        }
      }
      // check if one argument was not set. If so error
      for(int i=0; i<inputs.dim2(); i++)
        if(inputs.elem(i).is_bool_type()) // a bool type is an error (see above)
          throw runtime_error("All argXName attributes up to the largest argument number must be specified.");
    }
  
    // a XML vector
    ec=e->FirstChildElement(MBXMLUTILSPVNS"xmlVector");
    if(ec) {
      int i;
      // calculate nubmer for rows
      i=0;
      for(const TiXmlElement* ele=ec->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement(), i++);
      // get/eval values
      Matrix m(i, 1);
      octave_value octM;
      SXMatrixRef M;
      if(function) {
        octM=createSXMatrix("dummy", i, 1);
        M=cast<SXMatrixRef>(octM);
      }
      i=0;
      for(const TiXmlElement* ele=ec->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement(), i++)
        if(!function)
          m(i)=stringToOctValue(ele->GetText(), ele).double_value();
        else {
          SXMatrixRef Mele=cast<SXMatrixRef>(stringToOctValue(ele->GetText(), ele));
          if(Mele->size1()!=1 || Mele->size2()!=1) throw runtime_error("Scalar argument required.");
          M->elem(i,0)=Mele->elem(0,0);
        }
      if(!function)
        return m;
      else
        return createSXFunction(inputs, octM);
    }
  
    // a XML matrix
    ec=e->FirstChildElement(MBXMLUTILSPVNS"xmlMatrix");
    if(ec) {
      int i, j;
      // calculate nubmer for rows and cols
      i=0;
      for(const TiXmlElement* row=ec->FirstChildElement(); row!=0; row=row->NextSiblingElement(), i++);
      j=0;
      for(const TiXmlElement* ele=ec->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement(), j++);
      // get/eval values
      Matrix m(i, j);
      octave_value octM;
      SXMatrixRef M;
      if(function) {
        octM=createSXMatrix("dummy", i, j);
        M=cast<SXMatrixRef>(octM);
      }
      i=0;
      for(const TiXmlElement* row=ec->FirstChildElement(); row!=0; row=row->NextSiblingElement(), i++) {
        j=0;
        for(const TiXmlElement* col=row->FirstChildElement(); col!=0; col=col->NextSiblingElement(), j++)
          if(!function)
            m(j*m.rows()+i)=stringToOctValue(col->GetText(), col).double_value();
          else {
            SXMatrixRef Mele=cast<SXMatrixRef>(stringToOctValue(col->GetText(), col));
            if(Mele->size1()!=1 || Mele->size2()!=1) throw runtime_error("Scalar argument required.");
            M->elem(i,0)=Mele->elem(0,0);
          }
      }
      if(!function)
        return m;
      else
        return createSXFunction(inputs, octM);
    }
  
    // a element with a single text child (including unit conversion)
    if(e->GetText() && !e->FirstChildElement()) {
      octave_value ret=stringToOctValue(e->GetText(), e);
  
      // convert unit
      if(e->Attribute("unit") || e->Attribute("convertUnit")) {
        OctEval oe;
        oe.addParam("value", ret);
        if(e->Attribute("unit")) // convert with predefined unit
          ret=oe.stringToOctValue(units[e->Attribute("unit")], e);
        if(e->Attribute("convertUnit")) // convert with user defined unit
          ret=oe.stringToOctValue(e->Attribute("convertUnit"), e);
      }
  
      if(!function)
        return ret;
      else
        return createSXFunction(inputs, ret);
    }
  
    // rotation about x,y,z
    for(char ch='X'; ch<='Z'; ch++) {
      static octave_function *rotFunc[3]={
        symbol_table::find_function("rotateAboutX").function_value(), // get ones a pointer performance reasons
        symbol_table::find_function("rotateAboutY").function_value(), // get ones a pointer performance reasons
        symbol_table::find_function("rotateAboutZ").function_value()  // get ones a pointer performance reasons
      };
      ec=e->FirstChildElement(string(MBXMLUTILSPVNS"about")+ch);
      if(ec) {
        // check deprecated feature
        if(e->Attribute("unit")!=NULL)
          Deprecated::registerMessage("'unit' attribute for rotation matrix is no longer allowed.", e);
        // convert
        octave_value angle=eval(ec);
        octave_value_list ret=feval(rotFunc[ch-'X'], octave_value_list(angle), 1);
        if(error_state!=0) { error_state=0; throw runtime_error(string("Unable to generate rotation matrix using rotateAbout")+ch+"."); }
        return ret(0);
      }
    }
  
    // rotation cardan or euler
    for(int i=0; i<2; i++) {
      static string rotFuncName[2]={
        "cardan",
        "euler"
      };
      static octave_function *rotFunc[2]={
        symbol_table::find_function(rotFuncName[0]).function_value(), // get ones a pointer performance reasons
        symbol_table::find_function(rotFuncName[1]).function_value()  // get ones a pointer performance reasons
      };
      ec=e->FirstChildElement(string(MBXMLUTILSPVNS)+rotFuncName[i]);
      if(ec) {
        // check deprecated feature
        if(e->Attribute("unit")!=NULL)
          Deprecated::registerMessage("'unit' attribute for rotation matrix is no longer allowed.", e);
        // convert
        octave_value_list angles;
        const TiXmlElement *ele;
  
        ele=ec->FirstChildElement();
        angles.append(eval(ele));
        ele=ele->NextSiblingElement();
        angles.append(eval(ele));
        ele=ele->NextSiblingElement();
        angles.append(eval(ele));
        octave_value_list ret=feval(rotFunc[i], angles, 1);
        if(error_state!=0) { error_state=0; throw runtime_error(string("Unable to generate rotation matrix using ")+rotFuncName[i]); }
        return ret(0);
      }
    }
  
    // from file
    ec=e->FirstChildElement(MBXMLUTILSPVNS"fromFile");
    if(ec) {
      static octave_function *loadFunc=symbol_table::find_function("load").function_value();  // get ones a pointer performance reasons
      octave_value fileName=stringToOctValue(ec->Attribute("href"), ec);
      octave_value_list ret=feval(loadFunc, octave_value_list(fileName), 1);
      if(error_state!=0) { error_state=0; throw runtime_error(string("Unable to load file ")+ec->Attribute("href")); }
      return ret(0);
    }
  }
  
  // unknown element return a empty value
  return octave_value();
}

OctEval::ValueType OctEval::getType(const octave_value &value) {
  if(value.is_scalar_type() && value.is_real_type() && !value.is_string())
    return ScalarType;
  else if(value.is_matrix_type() && value.is_real_type() && !value.is_string()) {
    Matrix m=value.matrix_value();
    if(m.cols()==1)
      return VectorType;
    else
      return MatrixType;
  }
  else {
    // get the casadi type
    static octave_function *swig_type=symbol_table::find_function("swig_type").function_value(); // get ones a pointer to swig_type for performance reasons
    octave_value swigType;
    {
      BLOCK_STDERR;
      swigType=feval(swig_type, value, 1)(0);
    }
    if(error_state!=0) {
      error_state=0;
      if(value.is_string())
        return StringType;
      throw runtime_error("The provided octave value has an unknown type.");
    }
    if(swigType.string_value()=="SXMatrix")
      return SXMatrixType;
    else if(swigType.string_value()=="SXFunction")
      return SXFunctionType;
    else
      throw runtime_error("The provided octave value has an unknown type.");
  }
}

} // end namespace MBXMLUtils
