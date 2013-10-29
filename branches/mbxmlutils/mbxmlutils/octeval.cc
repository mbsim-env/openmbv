#include "mbxmlutils/octeval.h"
#include "mbxmlutilstinyxml/tinyxml.h"
#include <stdexcept>
#include <boost/filesystem.hpp>
#include "mbxmlutilstinyxml/tinynamespace.h"
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <mbxmlutilstinyxml/utils.h>
#include <octave/octave.h>
#include <boost/math/special_functions/round.hpp>
#include <casadi/symbolic/sx/sx_tools.hpp>

//MFMF: should also compile if casadi is not present; check throw statements

using namespace std;

namespace MBXMLUtils {

// Store the current directory in the ctor an restore in the dtor
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

NewParamLevel::NewParamLevel(OctEval &oe_, bool newLevel_) : oe(oe_), newLevel(newLevel_) {
  if(newLevel)
    oe.pushParams();
}

NewParamLevel::~NewParamLevel() {
  if(newLevel)
    oe.popParams();
}

template<>
string OctEval::cast<string>(const octave_value &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  ValueType type=getType(value);
  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(type==StringType) {
    ret<<"\""<<value.string_value()<<"\"";
    return ret.str();
  }
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXMatrixType || type==DMatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    if(type!=ScalarType)
      ret<<"[";
    for(int i=0; i<m.size(); i++) {
      for(int j=0; j<m[i].size(); j++) {
        long mlong=0;
        try {
           mlong=boost::math::lround(m[i][j]);
        }
        catch(...) {}
        double delta=fabs(mlong-m[i][j]);
        if(delta>eps*m[i][j] && delta>eps)
          ret<<m[i][j]<<(j<m[i].size()-1?",":"");
        else
          ret<<mlong<<(j<m[i].size()-1?",":"");
      }
      ret<<(i<m.size()-1?" ; ":"");
    }
    if(type!=ScalarType || m.size()==0)
      ret<<"]";
    return ret.str();
  }
  throw runtime_error("Can not convert this octave variable to a string.");
}

template<>
long OctEval::cast<long>(const octave_value &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType) {
    long ret=boost::math::lround(m[0][0]);
    double delta=fabs(ret-m[0][0]);
    if(delta>eps*m[0][0] && delta>eps)
      throw runtime_error("Canot cast octave value to long.");
    else
      return ret;
  }
  throw runtime_error("Cannot cast octave value to long.");
}

template<>
double OctEval::cast<double>(const octave_value &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType)
    return m[0][0];
  throw runtime_error("Cannot cast octave value to vector<double>.");
}

template<>
vector<double> OctEval::cast<vector<double> >(const octave_value &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType) {
    vector<double> ret;
    for(size_t i=0; i<m.size(); i++)
      ret.push_back(m[i][0]);
    return ret;
  }
  throw runtime_error("Cannot cast octave value to vector<double>.");
}

template<>
vector<vector<double> > OctEval::cast<vector<vector<double> > >(const octave_value &value) {
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > ret;
    Matrix m=value.matrix_value();
    for(int i=0; i<m.rows(); i++) {
      vector<double> row;
      for(int j=0; j<m.cols(); j++)
        row.push_back(m(j*m.rows()+i));
      ret.push_back(row);
    }
    return ret;
  }
  if(type==SXMatrixType || type==DMatrixType) {
    CasADi::SXMatrix m;
    if(type==SXMatrixType)
      m=cast<CasADi::SXMatrix>(value);
    else
      m=cast<CasADi::DMatrix>(value);
    if(!CasADi::isConstant(m))
      throw runtime_error("Can only cast constant value to vector<vector<double> >.");
    vector<vector<double> > ret;
    for(int i=0; i<m.size1(); i++) {
      vector<double> row;
      for(int j=0; j<m.size2(); j++)
        row.push_back(m.elem(i,j).getValue());
      ret.push_back(row);
    }
    return ret;
  }
  throw runtime_error("Cannot cast octave value to vector<vector<double> >.");
}

template<>
auto_ptr<TiXmlElement> OctEval::cast<auto_ptr<TiXmlElement> >(const octave_value &value) {
  if(getType(value)==SXFunctionType)
    return auto_ptr<TiXmlElement>(convertCasADiToXML(cast<CasADi::SXFunction>(value)));
  throw runtime_error("Cannot cast octave value to auto_ptr<TiXmlElement>.");
}

template<>
CasADi::SXMatrix OctEval::cast<CasADi::SXMatrix>(const octave_value &value) {
  ValueType type=getType(value);
  if(type==SXMatrixType)
    return castToSwig<CasADi::SXMatrix>(value);
  if(type==DMatrixType)
    return castToSwig<CasADi::DMatrix>(value);
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    CasADi::SXMatrix ret(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw runtime_error("Cannot cast to CasADi::SXMatrix*.");
}

template<>
CasADi::SXMatrix* OctEval::cast<CasADi::SXMatrix*>(const octave_value &value) {
  if(getType(value)==SXMatrixType)
    return castToSwig<CasADi::SXMatrix*>(value);
  throw runtime_error("Cannot cast to CasADi::SXMatrix*.");
}

template<>
CasADi::DMatrix OctEval::cast<CasADi::DMatrix>(const octave_value &value) {
  ValueType type=getType(value);
  if(type==DMatrixType)
    return castToSwig<CasADi::DMatrix>(value);
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXMatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    CasADi::DMatrix ret(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw runtime_error("Cannot cast to CasADi::DMatrix.");
}

template<>
CasADi::DMatrix* OctEval::cast<CasADi::DMatrix*>(const octave_value &value) {
  if(getType(value)==DMatrixType)
    return castToSwig<CasADi::DMatrix*>(value);
  throw runtime_error("Cannot cast to CasADi::DMatrix*.");
}

template<>
CasADi::SXFunction OctEval::cast<CasADi::SXFunction>(const octave_value &value) {
  if(getType(value)==SXFunctionType)
    return castToSwig<CasADi::SXFunction>(value);
  throw runtime_error("Cannot cast to CasADi::SXFunction.");
}

template<>
CasADi::SXFunction* OctEval::cast<CasADi::SXFunction*>(const octave_value &value) {
  if(getType(value)==SXFunctionType)
    return castToSwig<CasADi::SXFunction*>(value);
  throw runtime_error("Cannot cast to CasADi::SXFunction*.");
}

octave_value OctEval::createCasADi(const string &name) {
  static list<octave_value_list> idx;
  if(idx.empty()) {
    idx.push_back(octave_value_list(name));
    idx.push_back(octave_value_list());
  }
  *idx.begin()=octave_value_list(name);
  return casadiOctValue.subsref(".(", idx);
}

int OctEval::initCount=0;

std::map<std::string, std::string> OctEval::units;

octave_value OctEval::casadiOctValue;

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
      BLOCK_STDERR(blockstderr);
      casadiOctValue=feval("swigLocalLoad", octave_value_list("casadi"), 1)(0);
      if(error_state!=0) { error_state=0; throw string("Internal error: unable to initialize casadi."); }
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
        BLOCK_STDERR(blockstderr);
        ret=eval(*ee);
      }
      catch(const exception &ex) {
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
    vector<string> msg;
    msg.push_back("Error in one of the following parameters or infinit loop in this parameters:\n");
    for(list<const TiXmlElement*>::iterator ee=c.begin(); ee!=c.end(); ee++) {
      try {
        eval(*ee);
      }
      catch(const TiXmlException &ex) {
        msg.insert(msg.end(), ex.getMessage().begin(), ex.getMessage().end());
      }
      catch(const exception &ex) {
        msg.push_back(ex.what());
      }
    }
    msg.push_back("Error processing parameters. See above.");
    throw TiXmlException(msg);
  }
}

void OctEval::pushParams() {
  paramStack.push(currentParam);
}

void OctEval::popParams() {
  currentParam=paramStack.top();
  paramStack.pop();
}

void OctEval::addPath(const boost::filesystem::path &dir) {
  static octave_function *addpath=symbol_table::find_function("addpath").function_value();  // get ones a pointer performance reasons
  fevalThrow(addpath, octave_value_list(octave_value(dir.generic_string())));
}

octave_value OctEval::stringToOctValue(const std::string &str, const TiXmlElement *e) const {
  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  const TiXmlElement *base=TiXml_GetElementWithXmlBase(e, 0);
  if(base) // set working dir to path of current file, so that octave works with correct relative paths
    boost::filesystem::current_path(fixPath(base->Attribute("xml:base"), "."));

  // clear octave
  symbol_table::clear_variables();
  // restore current parameters
  for(map<string, octave_value>::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    symbol_table::varref(i->first)=i->second;

  ostringstream err;
  try{
    int dummy;
    BLOCK_STDOUT(blockstdout);
    REDIR_STDERR(redirstderr, err.rdbuf());
    eval_string(str, true, dummy, 0); // eval as statement list
  }
  catch(const TiXmlException &ex) {
    error_state=0;
    vector<string> msg;
    msg.push_back(err.str());
    msg.insert(msg.end(), ex.getMessage().begin(), ex.getMessage().end());
    throw TiXmlException(msg);
  }
  catch(const exception &ex) {
    error_state=0;
    throw TiXmlException(err.str()+ex.what(), e);
  }
  if(error_state!=0) { // if error => wrong code => throw error
    error_state=0;
    throw TiXmlException(err.str()+"Unable to evaluate expression: "+str, e);
  }
  // generate a strNoSpace from str by removing leading/trailing spaces as well as trailing ';'.
  string strNoSpace=str;
  while(strNoSpace.size()>0 && strNoSpace[0]==' ')
    strNoSpace=strNoSpace.substr(1);
  while(strNoSpace.size()>0 && (strNoSpace[strNoSpace.size()-1]==' ' || strNoSpace[strNoSpace.size()-1]==';'))
    strNoSpace=strNoSpace.substr(0, strNoSpace.size()-1);
  if(!symbol_table::is_variable("ret") && !symbol_table::is_variable("ans") && !symbol_table::is_variable(strNoSpace)) {
    throw TiXmlException("'ret' variable not defined in multi statement octave expression or incorrect single statement: "+
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
      string s=e->Attribute(attrName.c_str());
      size_t i;
      while((i=s.find('{'))!=string::npos) {
        size_t j=i;
        do {
          j=s.find('}', j+1);
          if(j==string::npos) throw TiXmlException("no matching } found in attriubte.", e);
        }
        while(s[j-1]=='\\'); // skip } which is quoted with backslash
        string evalStr=s.substr(i+1,j-i-1);
        // remove the backlash quote from { and }
        size_t k=0;
        while((k=evalStr.find('{', k))!=string::npos) {
          if(k==0 || evalStr[k-1]!='\\') throw TiXmlException("{ must be quoted with a backslash inside {...}.", e);
          evalStr=evalStr.substr(0, k-1)+evalStr.substr(k);
        }
        k=0;
        while((k=evalStr.find('}', k))!=string::npos) {
          if(k==0 || evalStr[k-1]!='\\') throw TiXmlException("} must be quoted with a backslash inside {...}.", e);
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
    NewParamLevel newParamLevel(*this, function);
    vector<CasADi::SXMatrix> inputs;
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

          octave_value octArg=createCasADi("SXMatrix");
          CasADi::SXMatrix *arg=cast<CasADi::SXMatrix*>(octArg);
          *arg=CasADi::ssym(a->Value(), dim, 1);
          addParam(a->Value(), octArg);
          inputs.resize(max(nr, static_cast<int>(inputs.size()))); // fill new elements with default ctor (isNull()==true)
          inputs[nr-1]=*arg;
        }
      }
      // check if one argument was not set. If so error
      for(int i=0; i<inputs.size(); i++)
        if(inputs[i].isNull()) // a isNull() object is a error (see above), since not all arg?name args were defined
          throw TiXmlException("All argXName attributes up to the largest argument number must be specified.", e);
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
      CasADi::SXMatrix M;
      if(function)
        M.resize(i, 1);
      i=0;
      for(const TiXmlElement* ele=ec->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement(), i++)
        if(!function)
          m(i)=stringToOctValue(ele->GetText(), ele).double_value();
        else {
          CasADi::SXMatrix Mele=cast<CasADi::SXMatrix>(stringToOctValue(ele->GetText(), ele));
          if(Mele.size1()!=1 || Mele.size2()!=1) throw TiXmlException("Scalar argument required.", e);
          M.elem(i,0)=Mele.elem(0,0);
        }
      if(!function)
        return handleUnit(e, m);
      else {
        octave_value octF=createCasADi("SXFunction");
        CasADi::SXFunction f(inputs, M);
        cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
        return octF;
      }
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
      CasADi::SXMatrix M;
      if(function)
        M.resize(i, j);
      i=0;
      for(const TiXmlElement* row=ec->FirstChildElement(); row!=0; row=row->NextSiblingElement(), i++) {
        j=0;
        for(const TiXmlElement* col=row->FirstChildElement(); col!=0; col=col->NextSiblingElement(), j++)
          if(!function)
            m(j*m.rows()+i)=stringToOctValue(col->GetText(), col).double_value();
          else {
            CasADi::SXMatrix Mele=cast<CasADi::SXMatrix>(stringToOctValue(col->GetText(), col));
            if(Mele.size1()!=1 || Mele.size2()!=1) throw TiXmlException("Scalar argument required.", e);
            M.elem(i,0)=Mele.elem(0,0);
          }
      }
      if(!function)
        return handleUnit(e, m);
      else {
        octave_value octF=createCasADi("SXFunction");
        CasADi::SXFunction f(inputs, M);
        cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
        return octF;
      }
    }
  
    // a element with a single text child (including unit conversion)
    if(e->GetText() && !e->FirstChildElement()) {
      octave_value ret=stringToOctValue(e->GetText(), e);
  
      // convert unit
      ret=handleUnit(e, ret);
  
      if(!function)
        return ret;
      else {
        octave_value octF=createCasADi("SXFunction");
        CasADi::SXFunction f(inputs, cast<CasADi::SXMatrix>(ret));
        cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
        return octF;
      }
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
        octave_value_list ret=fevalThrow(rotFunc[ch-'X'], octave_value_list(angle), 1, string("Unable to generate rotation matrix using rotateAbout")+ch+".", e);
        return ret(0);
      }
    }
  
    // rotation cardan or euler
    for(int i=0; i<2; i++) {
      static const string rotFuncName[2]={
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
        angles.append(handleUnit(ec, eval(ele)));
        ele=ele->NextSiblingElement();
        angles.append(handleUnit(ec, eval(ele)));
        ele=ele->NextSiblingElement();
        angles.append(handleUnit(ec, eval(ele)));
        octave_value_list ret=fevalThrow(rotFunc[i], angles, 1, string("Unable to generate rotation matrix using ")+rotFuncName[i], e);
        return ret(0);
      }
    }
  
    // from file
    ec=e->FirstChildElement(MBXMLUTILSPVNS"fromFile");
    if(ec) {
      static octave_function *loadFunc=symbol_table::find_function("load").function_value();  // get ones a pointer performance reasons
      octave_value fileName=stringToOctValue(ec->Attribute("href"), ec);

      // restore current dir on exit and change current dir
      PreserveCurrentDir preserveDir;
      const TiXmlElement *base=TiXml_GetElementWithXmlBase(e, 0);
      if(base) // set working dir to path of current file, so that octave works with correct relative paths
        boost::filesystem::current_path(fixPath(base->Attribute("xml:base"), "."));

      octave_value_list ret=fevalThrow(loadFunc, octave_value_list(fileName), 1, string("Unable to load file ")+ec->Attribute("href"), e);
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
      BLOCK_STDERR(blockstderr);
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
    else if(swigType.string_value()=="DMatrix")
      return DMatrixType;
    else if(swigType.string_value()=="SXFunction")
      return SXFunctionType;
    else
      throw runtime_error("The provided octave value has an unknown type.");
  }
}

octave_value_list OctEval::fevalThrow(octave_function *func, const octave_value_list &arg, int n,
                                      const string &msg, const TiXmlElement *e) {
  ostringstream err;
  octave_value_list ret;
  {
    REDIR_STDERR(redirstderr, err.rdbuf());
    ret=feval(func, arg, n);
  }
  if(error_state!=0) {
    error_state=0;
    if(!e)
      throw runtime_error(err.str()+msg);
    else
      throw TiXmlException(err.str()+msg, e);
  }
  return ret;
}

octave_value OctEval::handleUnit(const TiXmlElement *e, const octave_value &ret) {
  if(e->Attribute("unit") || e->Attribute("convertUnit")) {
    OctEval oe;
    oe.addParam("value", ret);
    if(e->Attribute("unit")) // convert with predefined unit
      return oe.stringToOctValue(units[e->Attribute("unit")], e);
    if(e->Attribute("convertUnit")) // convert with user defined unit
      return oe.stringToOctValue(e->Attribute("convertUnit"), e);
  }
  return ret;
}

} // end namespace MBXMLUtils
