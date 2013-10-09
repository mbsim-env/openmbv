#include <mbxmlutils/utils.h>
#include <mbxmlutils/octeval.h>
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <libxml/xmlschemas.h>
#include <libxml/xinclude.h>
#include <fstream>
#include <unistd.h>
#include <cmath>
#ifdef HAVE_UNORDERED_SET
#  include <unordered_set>
#else
#  include <set>
#  define unordered_set set
#endif
#include "mbxmlutilstinyxml/tinynamespace.h"
#include <boost/filesystem.hpp>
#ifdef _WIN32 // Windows
#  include "windows.h"
#endif
#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
#  include "mbxmlutilstinyxml/casadiXML.h"
#endif

using namespace std;
using namespace MBXMLUtils;
namespace bfs=boost::filesystem;

// a global octave evaluator
MBXMLUtils::OctaveEvaluator *octEval=NULL;

string SCHEMADIR;

void addFilesInDir(ostringstream &dependencies, string dir, string ext) {
  for(bfs::directory_iterator it=bfs::directory_iterator(dir.c_str()); it!=bfs::directory_iterator(); it++)
    if(it->path().extension().generic_string()==ext)
      dependencies<<it->path().generic_string()<<endl;
}

void warningCallback(void *ctx, const char *msg, ...) {
  // Schemas may be loaded from e.g aa/bb.xsd and the same schema from aa/../aa/bb.xsd
  // This is misleadingly interpreted on Windows as different schemas.
  // Hence we disable the corresponding warning (if not MBXMLUTILS_SCHEMAWARNINGS is set)
  static int printWarning=-1;
  if(printWarning==-1 && getenv("MBXMLUTILS_SCHEMAWARNINGS")!=NULL) printWarning=1; else printWarning=0;
  if(printWarning) {
    va_list ap;
    va_start(ap, msg);
    vfprintf(stderr, msg, ap);
    va_end(ap);
  }
}
// validate file using schema (currently by libxml)
int validate(const string &schema, const string &file) {
  cout<<"Parse and validate "<<file<<endl;

  // cache alread validated files
  static unordered_set<string> validatedCache;
  pair<unordered_set<string>::iterator, bool> ins2=validatedCache.insert(file);
  if(!ins2.second)
    return 0;

  xmlDoc *doc;
  doc=xmlReadFile(file.c_str(), NULL, XML_PARSE_XINCLUDE | XML_PARSE_NOXINCNODE | XML_PARSE_NOBASEFIX);
  if(!doc) return 1;
  if(xmlXIncludeProcess(doc)<0) return 1;

  // cache compiled schema
  xmlSchemaValidCtxtPtr compiledSchema=NULL;
  static unordered_map<string, xmlSchemaValidCtxtPtr> compiledSchemaCache;
  pair<unordered_map<string, xmlSchemaValidCtxtPtr>::iterator, bool> ins=compiledSchemaCache.insert(pair<string, xmlSchemaValidCtxtPtr>(schema, compiledSchema));
  if(ins.second) {
    xmlSchemaParserCtxt *schemaCtxt=xmlSchemaNewParserCtxt(schema.c_str());
    xmlSchemaSetParserErrors(schemaCtxt, NULL, &warningCallback, NULL); // redirect warning messages
    compiledSchema=ins.first->second=xmlSchemaNewValidCtxt(xmlSchemaParse(schemaCtxt));
  }
  else
    compiledSchema=ins.first->second;

  int ret=xmlSchemaValidateDoc(compiledSchema, doc); // validate using compiled/cached schema

  xmlFreeDoc(doc);
  if(ret!=0) return ret;
  return 0;
}

void embed(TiXmlElement *&e, const string &nslocation, map<string,string> &nsprefix, ostream &dependencies) {
  try {
    if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
      octEval->octavePushParams();
      // check if only href OR child element (This is not checked by the schema)
      TiXmlElement *l=0, *dummy;
      for(dummy=e->FirstChildElement(); dummy!=0; l=dummy, dummy=dummy->NextSiblingElement());
      if((e->Attribute("href") && l && l->ValueStr()!=MBXMLUTILSPVNS"localParameter") ||
         (e->Attribute("href")==0 && (l==0 || l->ValueStr()==MBXMLUTILSPVNS"localParameter"))) {
        TiXml_location(e, "", ": Only the href attribute OR a child element (expect pv:localParameter) is allowed in embed!");
        throw 1;
      }
      // check if attribute count AND counterName or none of both
      if((e->Attribute("count")==0 && e->Attribute("counterName")!=0) ||
         (e->Attribute("count")!=0 && e->Attribute("counterName")==0)) {
        TiXml_location(e, "", ": Only both, the count and counterName attribute must be given or none of both!");
        throw 1;
      }
  
      // get file name if href attribute exist
      string file="";
      if(e->Attribute("href")) {
        file=fixPath(TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base"), e->Attribute("href"));
        dependencies<<file<<endl;
      }
  
      // get onlyif attribute if exist
      string onlyif="1";
      if(e->Attribute("onlyif"))
        onlyif=e->Attribute("onlyif");
  
      // evaluate count using parameters
      int count=1;
      if(e->Attribute("count")) {
        octEval->octaveEvalRet(e->Attribute("count"), e);
        count=static_cast<int>(floor(octEval->octaveGetDoubleRet()+0.5));
      }
  
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(e->Attribute("counterName"))
        counterName=e->Attribute("counterName");
  
      TiXmlDocument *enewdoc=NULL;
      TiXmlElement *enew;
      // validate/load if file is given
      if(file!="") {
        if(validate(nslocation, file)!=0) {
          TiXml_location(e, "  included by: ", "");
          throw 1;
        }
        cout<<"Read "<<file<<endl;
        TiXmlDocument *enewdoc=new TiXmlDocument;
        enewdoc->LoadFile(file.c_str()); TiXml_PostLoadFile(enewdoc);
        enew=enewdoc->FirstChildElement();
        map<string,string> dummy;
        incorporateNamespace(enew, nsprefix, dummy, &dependencies);
      }
      else { // or take the child element (as a clone, because the embed element is deleted)
        if(e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter")
          enew=(TiXmlElement*)e->FirstChildElement()->NextSiblingElement()->Clone();
        else
          enew=(TiXmlElement*)e->FirstChildElement()->Clone();
        enew->SetAttribute("xml:base", TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base")); // add a xml:base attribute
      }
  
      // include a processing instruction with the line number of the original element
      TiXmlUnknown embedLine;
      embedLine.SetValue("?OriginalElementLineNr "+TiXml_itoa(e->Row())+"?");
      enew->InsertBeforeChild(enew->FirstChild(), embedLine);
  
  
      // generate local paramter for embed
      if(e->FirstChildElement() && e->FirstChildElement()->ValueStr()==MBXMLUTILSPVNS"localParameter") {
        // check if only href OR p:parameter child element (This is not checked by the schema)
        if((e->FirstChildElement()->Attribute("href") && e->FirstChildElement()->FirstChildElement()) ||
           (!e->FirstChildElement()->Attribute("href") && !e->FirstChildElement()->FirstChildElement())) {
          TiXml_location(e->FirstChildElement(), "", ": Only the href attribute OR the child element p:parameter) is allowed here!");
          throw 1;
        }
        cout<<"Generate local octave parameter string for "<<(file==""?"<inline element>":file)<<endl;
        if(e->FirstChildElement()->FirstChildElement()) // inline parameter
          octEval->fillParam(e->FirstChildElement()->FirstChildElement());
        else { // parameter from href attribute
          string paramFile=fixPath(TiXml_GetElementWithXmlBase(e,0)->Attribute("xml:base"), e->FirstChildElement()->Attribute("href"));
          // add local parameter file to dependencies
          dependencies<<paramFile<<endl;
          // validate local parameter file
          if(validate(SCHEMADIR+"/http___openmbv_berlios_de_MBXMLUtils/parameter.xsd", paramFile)!=0) {
            TiXml_location(e->FirstChildElement(), "  included by: ", "");
            throw 1;
          }
          // read local parameter file
          cout<<"Read local parameter file "<<paramFile<<endl;
          TiXmlDocument *localparamxmldoc=new TiXmlDocument;
          localparamxmldoc->LoadFile(paramFile.c_str()); TiXml_PostLoadFile(localparamxmldoc);
          TiXmlElement *localparamxmlroot=localparamxmldoc->FirstChildElement();
          map<string,string> dummy,dummy2;
          incorporateNamespace(localparamxmlroot,dummy,dummy2,&dependencies);
          // generate local parameters
          octEval->fillParam(localparamxmlroot);
          delete localparamxmldoc;
        }
      }
  
      // delete embed element and insert count time the new element
      for(int i=1; i<=count; i++) {
        // embed only if 'onlyif' attribute is true
        
        octEval->octaveAddParam(counterName, i);
        octEval->octaveEvalRet(onlyif, e);
        if(static_cast<int>(floor(octEval->octaveGetDoubleRet()+0.5))==1) {
          cout<<"Embed "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
          if(i==1) e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
          else e=(TiXmlElement*)(e->Parent()->InsertAfterChild(e, *enew));
  
          // include a processing instruction with the count number
          TiXmlUnknown countNr;
          countNr.SetValue("?EmbedCountNr "+TiXml_itoa(i)+"?");
          e->InsertAfterChild(e->FirstChild(), countNr);
      
          // apply embed to new element
          embed(e, nslocation, nsprefix, dependencies);
        }
        else
          cout<<"Skip embeding "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      if(enewdoc)
        delete enewdoc;
      else
        delete enew;
      octEval->octavePopParams();
      return;
    }
#ifdef HAVE_CASADI_SYMBOLIC_SX_SX_HPP
    else if(e->ValueStr()==MBXMLUTILSCASADINS"SXFunction")
      return; // skip processing of SXFunction elements
#endif
    else {
      octEval->eval(e);
    
      // THIS IS A WORKAROUND! Actually not all 'name' and 'ref*' attributes should be converted but only the
      // XML attributes of a type devived from pv:fullOctaveString and pv:partialOctaveString. But for that a
      // schema aware processor is needed!
      for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next())
        if(a->Name()==string("name") || string(a->Name()).substr(0,3)=="ref") {
          string s=a->ValueStr();
          int i;
          while((i=s.find('{'))>=0) {
            int j=s.find('}');
            octEval->octaveEvalRet(s.substr(i+1,j-i-1), e);
            s=s.substr(0,i)+octEval->octaveGetRet(MBXMLUtils::OctaveEvaluator::ScalarType)+s.substr(j+1);
          }
          a->SetValue(s);
        }
    }
  
    TiXmlElement *c=e->FirstChildElement();
    while(c) {
      embed(c, nslocation, nsprefix, dependencies);
      if(c==NULL) break;
      c=c->NextSiblingElement();
    }
  }
  catch(const string &str) {
    TiXml_location(e, "", ": "+str);
    throw 1;
  }
}

string extractFileName(const string& dirfilename) {
  bfs::path p(dirfilename.c_str());
  return p.filename().generic_string();
}

//////////////////////////MFMF
void walk(TiXmlElement *e, OctEval &oe) {
  cout<<e->ValueStr()<<endl;
  octave_value ret=oe.eval(e);
  try {
    CasADi::SXFunction f=OctEval::cast<CasADi::SXFunction>(ret);
    if(f.getNumInputs()==1) {
      if(f.inputExpr(0).size1()==1 && f.inputExpr(0).size2()==1) {
        f.init();
        f.setInput(CasADi::Matrix<double>(5.436), 0);
        f.evaluate();
        CasADi::SXMatrix out=f.output(0);
        for(int i=0; i<out.size1(); i++)
          for(int j=0; j<out.size2(); j++)
            cout<<"out "<<j<<" "<<i<<" "<<out(i,j)<<endl;
      }
    }
  }
  catch(...) {}
  if(e->Attribute("arg1name")) return;
  if(e->FirstChildElement())
    walk(e->FirstChildElement(), oe);
  if(e->NextSiblingElement())
    walk(e->NextSiblingElement(), oe);
}
//////////////////////////MFMF
#define PATHLENGTH 10240
int main(int argc, char *argv[]) {
  //////////////////////////MFMF
try
{
  OctEval oe;

  TiXmlDocument *paramxmldoc=new TiXmlDocument;
  paramxmldoc->LoadFile("/home/markus/project/mbsim/examples/xml/hierachical_modelling/parameter.mbsim.xml"); TiXml_PostLoadFile(paramxmldoc);
  TiXmlElement *e=paramxmldoc->FirstChildElement();
  map<string,string> dummy,dummy2;
  ostringstream dependencies;
  incorporateNamespace(e,dummy,dummy2,&dependencies);
  oe.addParamSet(e);
  delete paramxmldoc;

  paramxmldoc=new TiXmlDocument;
  paramxmldoc->LoadFile("/home/markus/project/mbsim/examples/xml/time_dependent_kinematics/parameter.mbsim.xml"); TiXml_PostLoadFile(paramxmldoc);
  e=paramxmldoc->FirstChildElement();
  incorporateNamespace(e,dummy,dummy2,&dependencies);
  oe.pushParams();
  oe.addParamSet(e);
  delete paramxmldoc;

  paramxmldoc=new TiXmlDocument;
  paramxmldoc->LoadFile("/home/markus/project/branch/svn/mbxmlutils/test.xml"); TiXml_PostLoadFile(paramxmldoc);
  e=paramxmldoc->FirstChildElement();
  incorporateNamespace(e,dummy,dummy2,&dependencies);
  walk(e, oe);
  delete paramxmldoc;
}  
catch(const OctEvalException &ex) {
  ex.print();
}
catch(const std::exception &ex) {
  cerr<<"ERROR: "<<ex.what()<<endl;
}
catch(...) {
  cerr<<"ERROR\n";
  return 1;
}
  return 0;
  //////////////////////////MFMF
  // convert argv to list
  list<string> arg;
  for(int i=1; i<argc; i++)
    arg.push_back(argv[i]);

  // help message
  if(arg.size()<3) {
    cout<<"Usage:"<<endl
        <<"mbxmlutilspp [--dependencies <dep-file-name>] [--mpath <dir> [--mpath <dir> ]]"<<endl
        <<"              <param-file> [dir/]<main-file> <namespace-location-of-main-file>"<<endl
        <<"             [<param-file> [dir/]<main-file> <namespace-location-of-main-file>]"<<endl
        <<"             ..."<<endl
        <<""<<endl
        <<"  --dependencies    Write a newline separated list of dependent files including"<<endl
        <<"                    <param-file> and <main-file> to <dep-file-name>"<<endl
        <<"  --mpath           Add <dir> to the octave search path for m-files"<<endl
        <<""<<endl
        <<"  The output file is named '.pp.<main-file>'."<<endl
        <<"  Use 'none' if not <param-file> is avaliabel."<<endl
        <<""<<endl
        <<"Copyright (C) 2009 Markus Friedrich <friedrich.at.gc@googlemail.com>"<<endl
        <<"This is free software; see the source for copying conditions. There is NO"<<endl
        <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
        <<""<<endl
        <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
    return 0;
  }

  // check for environment variables (none default installation)
  string XMLDIR;
  char *env;
  SCHEMADIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/schema";
  if((env=getenv("MBXMLUTILSSCHEMADIR"))) SCHEMADIR=env; // overwrite with envvar if exist
  XMLDIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/xml";
  if((env=getenv("MBXMLUTILSXMLDIR"))) XMLDIR=env; // overwrite with envvar if exist

  try {
    MBXMLUtils::OctaveEvaluator::initialize();
  
    // preserve whitespace and newline in TiXmlText nodes
    TiXmlBase::SetCondenseWhiteSpace(false);

    list<string>::iterator i, i2;
  
    // dependency file
    ostringstream dependencies;
    string depFileName="";
    if((i=std::find(arg.begin(), arg.end(), "--dependencies"))!=arg.end()) {
      i2=i; i2++;
      depFileName=(*i2);
      arg.erase(i); arg.erase(i2);
    }

    // mpath
    do {
      if((i=std::find(arg.begin(), arg.end(), "--mpath"))!=arg.end()) {
        i2=i; i2++;
        // the search path is global: use absolute path
        char curPath[PATHLENGTH];
        string absmpath=fixPath(string(getcwd(curPath, PATHLENGTH))+"/", *i2);
        // add to octave search path
        MBXMLUtils::OctaveEvaluator::addPath(absmpath);
        // add m-files in mpath dir to dependencies
        addFilesInDir(dependencies, *i2, ".m");
        arg.erase(i); arg.erase(i2);
      }
    }
    while(i!=arg.end());

    // loop over all files
    while(arg.size()>0) {
      // initialize the parameter stack (clear ALL caches)
      if(octEval) delete octEval;
      octEval=new MBXMLUtils::OctaveEvaluator;
  
      string paramxml=*arg.begin(); arg.erase(arg.begin());
      string mainxml=*arg.begin(); arg.erase(arg.begin());
      string nslocation=*arg.begin(); arg.erase(arg.begin());
  
      // validate parameter file
      if(paramxml!="none")
        if(validate(SCHEMADIR+"/http___openmbv_berlios_de_MBXMLUtils/parameter.xsd", paramxml)!=0) throw(1);
  
      // read parameter file
      TiXmlElement *paramxmlroot=NULL;
      if(paramxml!="none") {
        cout<<"Read "<<paramxml<<endl;
        TiXmlDocument *paramxmldoc=new TiXmlDocument;
        paramxmldoc->LoadFile(paramxml); TiXml_PostLoadFile(paramxmldoc);
        paramxmlroot=paramxmldoc->FirstChildElement();
        map<string,string> dummy,dummy2;
        dependencies<<paramxml<<endl;
        incorporateNamespace(paramxmlroot,dummy,dummy2,&dependencies);
      }
  
      // THIS IS A WORKAROUND! See before.
      // get units
      cout<<"Build unit list for measurements"<<endl;
      TiXmlDocument *mmdoc=new TiXmlDocument;
      mmdoc->LoadFile(XMLDIR+"/measurement.xml"); TiXml_PostLoadFile(mmdoc);
      TiXmlElement *ele, *el2;
      map<string,string> units;
      for(ele=mmdoc->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement())
        for(el2=ele->FirstChildElement(); el2!=0; el2=el2->NextSiblingElement()) {
          if(units.find(el2->Attribute("name"))!=units.end()) {
            cerr<<"ERROR! Unit name "<<el2->Attribute("name")<<" is defined more than once."<<endl;
            throw(1);
          }
          units[el2->Attribute("name")]=el2->GetText();
        }
      delete mmdoc;
      octEval->setUnits(units);
  
      // generate octave parameter string
      if(paramxml!="none") {
        cout<<"Generate octave parameter string from "<<paramxml<<endl;
        octEval->fillParam(paramxmlroot);
      }
  
      // validate main file
      if(validate(nslocation, mainxml)!=0) throw(1);
  
      // read main file
      cout<<"Read "<<mainxml<<endl;
      TiXmlDocument *mainxmldoc=new TiXmlDocument;
      mainxmldoc->LoadFile(mainxml); TiXml_PostLoadFile(mainxmldoc);
      TiXmlElement *mainxmlroot=mainxmldoc->FirstChildElement();
      map<string,string> nsprefix, dummy;
      dependencies<<mainxml<<endl;
      incorporateNamespace(mainxmlroot,nsprefix,dummy,&dependencies);
  
      // embed/validate/toOctave/unit/eval files
      embed(mainxmlroot, nslocation,nsprefix,dependencies);
  
      // save result file
      cout<<"Save preprocessed file "<<mainxml<<" as .pp."<<extractFileName(mainxml)<<endl;
      TiXml_addLineNrAsProcessingInstruction(mainxmlroot);
      unIncorporateNamespace(mainxmlroot, nsprefix);
      mainxmldoc->SaveFile(".pp."+string(extractFileName(mainxml)));
      delete mainxmldoc;
  
      // validate preprocessed file
      if(validate(nslocation, ".pp."+string(extractFileName(mainxml)))!=0) throw(1);
    }
  
    // output dependencies?
    if(depFileName!="") {
      ofstream dependenciesFile(depFileName.c_str());
      dependenciesFile<<dependencies.str();
      dependenciesFile.close();
    }
  }
  // do_octave_atexit must be called on error and no error before leaving
  catch(const string &str) {
    cerr<<"Exception: "<<str<<endl;
    MBXMLUtils::OctaveEvaluator::terminate();
    return 1;
  }
  catch(...) {
    cerr<<"Unknown exception."<<endl;
    MBXMLUtils::OctaveEvaluator::terminate();
    return 1;
  }
  MBXMLUtils::OctaveEvaluator::terminate();
  return 0;
}
