#include "mbxmlutils/utils.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "mbxmlutilstinyxml/tinynamespace.h"
#include <octave/parse.h>
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <mbxmlutilstinyxml/utils.h>
#include <octave/octave.h>
#include <octave/toplev.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <mbxmlutilstinyxml/casadiXML.h>
#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
#include "mbxmlutilstinyxml/casadiXML.h"
#endif

#define MBXMLUTILSPARAMNS_ "http://openmbv.berlios.de/MBXMLUtils/parameter"
#define MBXMLUTILSPARAMNS "{"MBXMLUTILSPARAMNS_"}"

//MFMF delete this file (MORE or less)

using namespace std;

namespace MBXMLUtils {

#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
static CasADi::SXMatrix getCasADiSXMatrixFromOctave(const string &varname);
#endif

template<int T>
class Block_ {
  public:
    Block_(ostream &str_) : str(str_) {
      if(disableCount==0)
        orgcxxx=str.rdbuf(0);
      disableCount++;
    }
    ~Block_() {
      disableCount--;
      if(disableCount==0)
        str.rdbuf(orgcxxx);
    }
  private:
    ostream &str;
    static streambuf *orgcxxx;
    static int disableCount;
};
template<int T> streambuf *Block_<T>::orgcxxx;
template<int T> int Block_<T>::disableCount=0;
#define BLOCK_STDOUT Block_<1> mbxmlutils_dummy_blockstdout(std::cout);
#define BLOCK_STDERR Block_<2> mbxmlutils_dummy_blockstderr(std::cerr);

class PreserveCurrentDir_ {
  public:
    PreserveCurrentDir_() {
      dir=boost::filesystem::current_path();
    }
    ~PreserveCurrentDir_() {
      boost::filesystem::current_path(dir);
    }
  private:
    boost::filesystem::path dir;
};



OctaveEvaluator::OctaveEvaluator() {
  symbol_table::clear_variables();
}

// add a parameter to the parameter list (used by octavePushParam and octavePopParams)
void OctaveEvaluator::octaveAddParam(const string &paramName, const octave_value& value, bool useCache) {
  // add paramter to parameter list if a parameter of the same name dose not exist in the list
  currentParam[paramName]=value;
  if(useCache) {
    if(paramStack.size()>=currentParamHash.size()) {
      currentParamHash.resize(paramStack.size()+1);
      currentParamHash[paramStack.size()]=0;
    }
    currentParamHash[paramStack.size()]++;
  }
}

void OctaveEvaluator::octaveAddParam(const string &paramName, double value, bool useCache) {
  octaveAddParam(paramName, octave_value(value), useCache);
}

// push all parameters from list to a parameter stack
void OctaveEvaluator::octavePushParams() {
  paramStack.push(currentParam);
}

// pop all parameters from list from the parameter stack
void OctaveEvaluator::octavePopParams() {
  // restore previous parameter list
  currentParam=paramStack.top();
  paramStack.pop();
}

// Evaluate a single statement or a statement list and save the result in the variable 'ret'
// If the resulting type is a scalar, vector, matrix or string NULL is returned.
// If the resulting type is a CasADi expression a XML representation of the expression is returned.
TiXmlElement* OctaveEvaluator::octaveEvalRet(string str, TiXmlElement *e, bool useCache, vector<pair<string, int> > *arg) {
  // restore current dir on exit
  PreserveCurrentDir_ preserveDir;

#if ! defined HAVE_CASADI_SYMBOLIC_SX_SX_HPP
  if(arg) throw string("Found a CasADi expression but MBXMLUtils was not compiled with CasADi support.");
#endif

  // disable the cache for casadi expressions
  if(arg) useCache=false;

  string id;
  if(useCache) {
    // a cache: this cache is only unique per input file on the command line.
    // Hence this cache must be cleared after/before each input file. Hence this cache is moved to a global variable.
    if(paramStack.size()>=currentParamHash.size()) {
      currentParamHash.resize(paramStack.size()+1);
      currentParamHash[paramStack.size()]=0;
    }
    stringstream s; s<<paramStack.size()<<";"<<currentParamHash[paramStack.size()]<<";"; id=s.str();
    unordered_map<string, octave_value>::iterator it=cache.find(id+str);
    if(it!=cache.end()) {
      symbol_table::varref("ret")=it->second;
      return NULL;
    }
  }

  // clear octave
  symbol_table::clear_variables();
  // restore current parameters
  for(map<string, octave_value>::iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    symbol_table::varref(i->first)=i->second;

  int dummy;

#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
  if(arg) {
    // initialize casadi if a casadi expresion should be processed
    {
      static octave_function *casadi=symbol_table::find_function("casadi").function_value(); // get ones a pointer to casadi for performance reasons
      BLOCK_STDERR;
      feval(casadi);
      if(error_state!=0) { error_state=0; throw string("Internal error: unable to initialize casadi."); }
    }

    // fill octave with the casadi inputs provided by arg
    for(size_t i=0; i<arg->size(); i++) {
      // MISSING: try to convert this eval_string(...) into pure octave API function call for speedup:
      // e.g. for like this (but this code is not working currently):
      // list<octave_value_list> subsArg;
      // subsArg.push_back(octave_value_list(octave_value("ssym")));
      // octave_function *casadi_ssym=symbol_table::varref("casadi").subsref(".", subsArg).function_value();
      // octave_value_list ssymArg;
      // ssymArg.append((*arg)[i].first);
      // ssymArg.append((*arg)[i].second);
      // ssymArg.append(1);
      // symbol_table::varref((*arg)[i].first)=feval(casadi_ssym, ssymArg, 1)(0);
      stringstream str;
      str<<(*arg)[i].first<<"=casadi.ssym('"<<(*arg)[i].first<<"', "<<(*arg)[i].second<<", 1);";
      eval_string(str.str(),true,dummy,0);
      if(error_state!=0) { error_state=0; throw string("Internal error: unable to fill the casadi inputs."); }
    }
  }
#endif

  const TiXmlElement *base=TiXml_GetElementWithXmlBase(e,0);
  if(base) // set working dir to path of current file, so that octave works with correct relative paths
    boost::filesystem::current_path(fixPath(base->Attribute("xml:base"),"."));

  try{
    BLOCK_STDOUT;
    eval_string(str,true,dummy,0); // eval as statement list
  }
  catch(...) {
    error_state=1;
  }
  if(error_state!=0) { // if error => wrong code => throw error
    error_state=0;
    throw string("Error in octave expression: "+str);
  }
  // generate a strNoSpace from str by removing leading/trailing spaces as well as trailing ';'.
  string strNoSpace=str;
  while(strNoSpace.size()>0 && strNoSpace[0]==' ')
    strNoSpace=strNoSpace.substr(1);
  while(strNoSpace.size()>0 && (strNoSpace[strNoSpace.size()-1]==' ' || strNoSpace[strNoSpace.size()-1]==';'))
    strNoSpace=strNoSpace.substr(0, strNoSpace.size()-1);
  if(!symbol_table::is_variable("ret") && !symbol_table::is_variable("ans") && !symbol_table::is_variable(strNoSpace)) {
    throw string("'ret' variable not defined in octave statement list or no single statement in: "+str);
  }
  if(symbol_table::is_variable(strNoSpace))
    symbol_table::varref("ret")=symbol_table::varref(strNoSpace);
  else if(!symbol_table::is_variable("ret"))
    symbol_table::varref("ret")=symbol_table::varref("ans");

  if(useCache)
    cache.insert(make_pair(id+str, symbol_table::varref("ret")));

#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
  // return casadi XML representation
  if(arg) {
    // get casadi inputs from octave
    vector<CasADi::SXMatrix> input;
    for(size_t i=0; i<arg->size(); i++)
      input.push_back(getCasADiSXMatrixFromOctave((*arg)[i].first));
    // get casadi output from octave
    CasADi::SXMatrix output=getCasADiSXMatrixFromOctave("ret");
    // create casadi SXFunction using inputs/output
    CasADi::SXFunction f(input, output);
    // return the XML representation of the SXFunction
    return CasADi::convertCasADiToXML(f);
  }
#endif

  return NULL;
}

void OctaveEvaluator::checkType(const octave_value& val, ValueType expectedType) {
  ValueType type;
  // get type of val
  if(val.is_scalar_type() && val.is_real_type() && (val.is_string()!=1))
    type=ScalarType;
  else if(val.is_matrix_type() && val.is_real_type() && (val.is_string()!=1)) {
    Matrix m=val.matrix_value();
    type=m.cols()==1?VectorType:MatrixType;
  }
  else if(val.is_string())
    type=StringType;
  else // throw on unknown type
    throw(string("Unknown type: none of scalar, vector, matrix or string"));
  // check for correct type
  if(expectedType!=ArbitraryType) {
    if(type==ScalarType && expectedType==StringType)
      throw string("Got scalar value, while a string is expected");
    if(type==VectorType && (expectedType==StringType || expectedType==ScalarType)) 
      throw string("Got column-vector value, while a ")+(expectedType==StringType?"string":"scalar")+" is expected";
    if(type==MatrixType && (expectedType==StringType || expectedType==ScalarType || expectedType==VectorType))
      throw string("Got matrix value, while a ")+(expectedType==StringType?"string":(expectedType==ScalarType?"scalar":"column-vector"))+" is expected";
    if(type==StringType && (expectedType==MatrixType || expectedType==ScalarType || expectedType==VectorType))
      throw string("Got string value, while a ")+(expectedType==MatrixType?"matrix":(expectedType==ScalarType?"scalar":"column-vector"))+" is expected";
  }
}

// return the value of 'ret'
string OctaveEvaluator::octaveGetRet(ValueType expectedType) {
  octave_value &o=symbol_table::varref("ret"); // get 'ret'

  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(o.is_scalar_type() && o.is_real_type() && (o.is_string()!=1)) {
    ret<<o.double_value();
  }
  else if(o.is_matrix_type() && o.is_real_type() && (o.is_string()!=1)) {
    Matrix m=o.matrix_value();
    ret<<"[";
    for(int i=0; i<m.rows(); i++) {
      for(int j=0; j<m.cols(); j++)
        ret<<m(j*m.rows()+i)<<(j<m.cols()-1?",":"");
      ret<<(i<m.rows()-1?" ; ":"]");
    }
    if(m.rows()==0)
      ret<<"]";
  }
  else if(o.is_string()) {
    ret<<"\""<<o.string_value()<<"\"";
  }
  else { // if not scalar, matrix or string => error
    throw(string("Unknown type: none of scalar, vector, matrix or string"));
  }
  checkType(o, expectedType);

  return ret.str();
}

double OctaveEvaluator::octaveGetDoubleRet() {
  octave_value v=symbol_table::varref("ret");
  checkType(v, MBXMLUtils::OctaveEvaluator::ScalarType);
  return v.double_value();
}

octave_value& OctaveEvaluator::octaveGetOctaveValueRet() {
  return symbol_table::varref("ret");
}

void OctaveEvaluator::fillParam(TiXmlElement *e, bool useCache) {
  // outer loop to resolve recursive parameters
  size_t length=0;
  for(TiXmlElement *ee=e->FirstChildElement(); ee!=NULL; ee=ee->NextSiblingElement())
    length++;
  for(size_t outerLoop=0; outerLoop<length; outerLoop++) {
    // evaluate parameter
    TiXmlElement *ee=e->FirstChildElement();
    while(ee!=NULL) {
      int err=0;
      octave_value ret;
      {
        BLOCK_STDERR;
        try { 
          eval(ee, useCache);
          ret=symbol_table::varref("ret");
          checkType(ret, ee->ValueStr()==MBXMLUTILSPARAMNS"scalarParameter"?ScalarType:
                         ee->ValueStr()==MBXMLUTILSPARAMNS"vectorParameter"?VectorType:
                         ee->ValueStr()==MBXMLUTILSPARAMNS"matrixParameter"?MatrixType:
                         ee->ValueStr()==MBXMLUTILSPARAMNS"stringParameter"?StringType:ArbitraryType);
        }
        catch(...) { err=1; }
      }
      if(err==0) { // if no error
        octaveAddParam(ee->Attribute("name"), ret, useCache); // add param to list
        TiXmlElement *eee=ee->NextSiblingElement();
        e->RemoveChild(ee);
        ee=eee;
      }
      else
        ee=ee->NextSiblingElement();
    }
  }
  if(e->FirstChildElement()!=NULL) { // if parameters are left => error
    cerr<<"Error in one of the following parameters or infinit loop in this parameters:\n";
    for(TiXmlElement *ee=e->FirstChildElement(); ee!=0; ee=ee->NextSiblingElement()) {
      try {
        eval(ee, useCache);
        octave_value ret=symbol_table::varref("ret");
        checkType(ret, ee->ValueStr()==MBXMLUTILSPARAMNS"scalarParameter"?ScalarType:
                       ee->ValueStr()==MBXMLUTILSPARAMNS"vectorParameter"?VectorType:
                       ee->ValueStr()==MBXMLUTILSPARAMNS"matrixParameter"?MatrixType:
                       ee->ValueStr()==MBXMLUTILSPARAMNS"stringParameter"?StringType:ArbitraryType);
      }
      catch(const string &str) { cerr<<str<<endl; }
      TiXml_location(ee, "", string(": Variable name: ")+ee->Attribute("name")); // output location of element
    }
    throw string("Erroring processing parameters. See above.");
  }
}

void OctaveEvaluator::saveAndClearCurrentParam() {
  savedCurrentParam=currentParam; // save parameters
  currentParam.clear(); // clear parameters
}

void OctaveEvaluator::restoreCurrentParam() {
  currentParam=savedCurrentParam; // restore parameter
}

static std::vector<char*> octave_argv;

void OctaveEvaluator::initialize() {
  struct stat st;
  char *env;
  string OCTAVEDIR;
  OCTAVEDIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/octave";
  if((env=getenv("MBXMLUTILSOCTAVEDIR"))) OCTAVEDIR=env; // overwrite with envvar if exist

  // OCTAVE_HOME
  string OCTAVE_HOME; // the string for putenv must has program life time
  if(getenv("OCTAVE_HOME")==NULL && stat((MBXMLUtils::getInstallPath()+"/share/octave").c_str(), &st)==0) {
    OCTAVE_HOME="OCTAVE_HOME="+MBXMLUtils::getInstallPath();
    putenv((char*)OCTAVE_HOME.c_str());
  }

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
  if(error_state!=0) { error_state=0; throw string("Internal error: unable to disable warnings."); }

  feval("addpath", octave_value_list(octave_value(OCTAVEDIR)));
  if(error_state!=0) { error_state=0; throw string("Internal error: cannot add octave search path."); }

#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
  // initialize casadi in octave
  feval("addpath", octave_value_list(octave_value(MBXMLUtils::getInstallPath()+"/bin")));
  if(error_state!=0) { error_state=0; throw string("Internal error: cannot add casadi octave search path."); }
#endif
}

void OctaveEvaluator::terminate() {
  do_octave_atexit();
}

void OctaveEvaluator::addPath(const std::string &path) {
  feval("addpath", octave_value_list(octave_value(path)));
  if(error_state!=0) { error_state=0; throw string("Error: cannot add octave search path '")+path+"'."; }
}

void OctaveEvaluator::eval(TiXmlElement *e, bool useCache) {
  TiXmlElement *c=e->FirstChildElement();

  // symbolic function
  // THIS IS A WORKAROUND! Actually not all Elements with a attribute named arg1name should be
  // converted but only elements with a attribute of type symbolicFunctionArgNameType.
  if(e->Attribute("arg1name")) {
    vector<pair<string, int> > arg;
    // loop over all attributes and search for arg1name,  arg2name attributes
    for(TiXmlAttribute *a=e->FirstAttribute(); a!=NULL; a=a->Next()) {
      string value=a->Name();
      if(value.substr(0,3)=="arg" && value.substr(value.length()-4,4)=="name") {
        int nr=atoi(value.substr(3, value.length()-4-3).c_str());
        int dim=1;
        stringstream str;
        str<<"arg"<<nr<<"dim";
        if(e->Attribute(str.str())) dim=atoi(e->Attribute(str.str().c_str()));
        arg.resize(nr);
        arg[nr-1]=make_pair(a->Value(), dim);
      }
    }
    // handle xml vector
    if(e->FirstChildElement(MBXMLUTILSPVNS"xmlVector")) {
      TiXmlElement *exp=NULL;
      TiXmlElement *matrix=NULL;
      for(TiXmlElement* ele=e->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        TiXmlElement *expXML=octaveEvalRet(ele->GetText(), ele, true, &arg);
        if(exp==NULL) {
          exp=expXML->Clone()->ToElement();
          exp->FirstChildElement(MBXMLUTILSCASADINS"outputs")->RemoveChild(
            exp->FirstChildElement(MBXMLUTILSCASADINS"outputs")->FirstChildElement());
          matrix=new TiXmlElement(MBXMLUTILSCASADINS"SXMatrix");
          exp->FirstChildElement(MBXMLUTILSCASADINS"outputs")->LinkEndChild(matrix);
          matrix->SetAttribute("columnVector", "true");
        }
        matrix->LinkEndChild(expXML->FirstChildElement(MBXMLUTILSCASADINS"outputs")->FirstChildElement()->Clone());
        delete expXML;
      }
      // remove xmlVector and add CasADi XML representation
      e->RemoveChild(e->FirstChild());
      e->LinkEndChild(exp);
    }
    // handle xml matrix
    else if(e->FirstChildElement(MBXMLUTILSPVNS"xmlMatrix")) {
      throw string("Matrix symbolic functions in XML notation are currently not implemented.");
    }
    // handle scalar
    else {
      // evaluate the expression and get XML representation as return value (use the provided arguments arg)
      TiXmlElement *expXML=octaveEvalRet(e->GetText(), e, true, &arg);
      // remove text child and add CasADi XML representation
      e->RemoveChild(e->FirstChild());
      e->LinkEndChild(expXML);
    }
    return;
  }

  // xmlMatrix
  if(c && c->ValueStr()==MBXMLUTILSPVNS"xmlMatrix") {
    string mat="[";
    for(TiXmlElement* row=c->FirstChildElement(); row!=0; row=row->NextSiblingElement()) {
      for(TiXmlElement* ele=row->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        eval(ele, useCache);
        mat+=ele->GetText();
        if(ele->NextSiblingElement()) mat+=",";
      }
      if(row->NextSiblingElement()) mat+=";\n";
    }
    mat+="]";
    TiXmlText text(mat);
    c->Parent()->InsertEndChild(text);
    c->Parent()->RemoveChild(c);
    return;
  }

  // xmlVector
  if(c && c->ValueStr()==MBXMLUTILSPVNS"xmlVector") {
    string vec="[";
    for(TiXmlElement* ele=c->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
      eval(ele, useCache);
      vec+=ele->GetText();
      if(ele->NextSiblingElement()) vec+=";";
    }
    vec+="]";
    TiXmlText text(vec);
    c->Parent()->InsertEndChild(text);
    c->Parent()->RemoveChild(c);
    eval(c, useCache);
    return;
  }

  // rotation about x,y,z
  for(char ch='X'; ch<='Z'; ch++)
    if(c && c->ValueStr()==string(MBXMLUTILSPVNS"about")+ch) {
      // check deprecated feature
      if(c->Parent()->ToElement()->Attribute("unit")!=NULL)
        Deprecated::registerMessage("'unit' attribute for rotation matrix is no longer allowed.",
                                    c->Parent()->ToElement());
      // convert
      c->SetValue("dummy"); // modify the element name to avoid a recursive call on the next line
      eval(c, useCache);
      TiXmlText text(string("rotateAbout")+ch+"("+c->GetText()+")");
      e->InsertEndChild(text);
      e->RemoveChild(c);
      eval(e, useCache);
      return;
    }

  // rotation cardan or euler
  string rotFkt[]={"cardan", "euler"};
  for(int i=0; i<2; i++)
    if(c && c->ValueStr()==MBXMLUTILSPVNS+rotFkt[i]) {
      // check deprecated feature
      if(c->Parent()->ToElement()->Attribute("unit")!=NULL)
        Deprecated::registerMessage("'unit' attribute for rotation matrix is no longer allowed.",
                                    c->Parent()->ToElement());
      // convert
      TiXmlElement *ele;
      string octaveStr=rotFkt[i]+"(";
      ele=c->FirstChildElement();
      eval(ele, useCache);
      octaveStr+=string(ele->GetText())+", ";
      ele->NextSiblingElement();
      eval(ele, useCache);
      octaveStr+=string(ele->GetText())+", ";
      ele->NextSiblingElement();
      eval(ele, useCache);
      octaveStr+=ele->GetText();
      octaveStr+=")";
      TiXmlText text(octaveStr);
      e->InsertEndChild(text);
      e->RemoveChild(c);
      eval(e, useCache);
      return;
    }

  // from file
  if(c && c->ValueStr()==MBXMLUTILSPVNS"fromFile") {
    octaveEvalRet(c->Attribute("href"), c, useCache);
    string loadStr("ret=load(");
    loadStr+=octaveGetRet(MBXMLUtils::OctaveEvaluator::StringType);
    loadStr+=");";
    TiXmlText text(loadStr);
    TiXmlElement *p=c->Parent()->ToElement();
    p->InsertEndChild(text);
    p->RemoveChild(c);
    eval(p, useCache);
    return;
  }

  // octave string with unit
  // THIS IS A WORKAROUND! Actually not all Text-Elements should be converted but only the Text-Elements
  // of XML elementx of a type devived from pv:scalar, pv:vector, pv:matrix and pv:string. But for that a
  // schema aware processor is needed!
  if(e->GetText() && e->FirstChildElement()==NULL) {
    // eval text node
    octaveEvalRet(e->GetText(), e, useCache);
    // convert unit
    if(e->Attribute("unit") || e->Attribute("convertUnit")) {
      saveAndClearCurrentParam();
      octaveAddParam("value", octaveGetOctaveValueRet(), useCache); // add 'value=ret', since unit-conversion used 'value'
      if(e->Attribute("unit")) { // convert with predefined unit
        octaveEvalRet(units[e->Attribute("unit")], NULL, useCache);
        e->RemoveAttribute("unit");
      }
      if(e->Attribute("convertUnit")) { // convert with user defined unit
        octaveEvalRet(e->Attribute("convertUnit"), NULL, useCache);
        e->RemoveAttribute("convertUnit");
      }
      restoreCurrentParam();
    }
    // write eval to xml
    e->FirstChild()->SetValue(octaveGetRet());
    e->FirstChild()->ToText()->SetCDATA(false);
    return;
  }
}

#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
// get from octave the variable named varname and extract the casadi object from it
// and return it as a SXMatrix object.
CasADi::SXMatrix getCasADiSXMatrixFromOctave(const string &varname) {
  BLOCK_STDOUT;
  BLOCK_STDERR;

  // get the octave symbol
  octave_value &var=symbol_table::varref(varname);
  // get the casadi type SX or SXMatrix
  static octave_function *swig_type=symbol_table::find_function("swig_type").function_value(); // get ones a pointer to swig_type for performance reasons
  octave_value swigType=feval(swig_type, var, 1)(0);
  if(error_state!=0) {
    error_state=0;
    // not a swig variable => constant
    const octave_value &val=OctaveEvaluator::octaveGetOctaveValueRet();
    if(val.is_scalar_type() && val.is_real_type() && (val.is_string()!=1)) {
      CasADi::SX ret(val.double_value());
      return ret;
    }
    else if(val.is_matrix_type() && val.is_real_type() && (val.is_string()!=1)) {
      Matrix m=val.matrix_value();
      CasADi::SXMatrix mat(m.rows(), m.cols());
      for(int i=0; i<m.rows(); i++)
        for(int j=0; j<m.cols(); j++)
          mat[i][j]=m(j*m.rows()+i);
      return mat;
    }
    else // throw on unknown type
      throw(string("Unknown type: must be a scalar, vector or matrix"));
  }
  bool sxMatrix=false;
  if(swigType.string_value()=="SX")
    sxMatrix=false;
  else if(swigType.string_value()=="SXMatrix")
    sxMatrix=true;
  // get the casadi pointer: octave returns a 32 or 64bit integer which represents the pointer
  static octave_function *swig_this=symbol_table::find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  octave_value swigThis=feval(swig_this, var, 1)(0);
  if(error_state!=0) { error_state=0; throw string("Internal error: unable to get the swig pointer."); }
  void *swigPtr=NULL;
  if(swigThis.is_uint64_type())
    swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  else if(swigThis.is_uint32_type())
    swigPtr=reinterpret_cast<void*>(swigThis.uint32_scalar_value().value());
  // convert the void pointer to the correct casadi type
  // (if it is a SX it is implicitly convert to SXMatrix)
  CasADi::SXMatrix ret;
  if(sxMatrix)
    ret=*static_cast<CasADi::SXMatrix*>(swigPtr);
  else
    ret=*static_cast<CasADi::SX*>(swigPtr);
  // return the casadi SXMatrix
  return ret;
}
#endif

} // end namespace MBXMLUtils
