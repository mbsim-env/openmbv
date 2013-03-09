#include <mbxmlutils/utils.h>
#include <mbxmlutilstinyxml/getinstallpath.h>
#include <libxml/xmlschemas.h>
#include <libxml/xinclude.h>
#include <fstream>
#include <unistd.h>
#ifdef HAVE_UNORDERED_SET
#  include <unordered_set>
#else
#  include <set>
#  define unordered_set set
#endif
#include "mbxmlutilstinyxml/tinynamespace.h"
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>
#include "env.h"
#include <boost/filesystem.hpp>
#ifdef _WIN32 // Windows
#  include "windows.h"
#endif

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

using namespace std;
namespace bfs=boost::filesystem;

// a global octave evaluator
MBXMLUtils::OctaveEvaluator *octEval=NULL;

string SCHEMADIR;

void addFilesInDir(ostringstream &dependencies, string dir, string ext) {
  for(bfs::directory_iterator it=bfs::directory_iterator(dir.c_str()); it!=bfs::directory_iterator(); it++)
    if(it->path().extension().generic_string()==ext)
      dependencies<<it->path().generic_string()<<endl;
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
  doc=xmlParseFile(file.c_str());
  if(!doc) return 1;
  if(xmlXIncludeProcess(doc)<0) return 1;

  // cache compiled schema
  xmlSchemaValidCtxtPtr compiledSchema=NULL;
  static unordered_map<string, xmlSchemaValidCtxtPtr> compiledSchemaCache;
  pair<unordered_map<string, xmlSchemaValidCtxtPtr>::iterator, bool> ins=compiledSchemaCache.insert(pair<string, xmlSchemaValidCtxtPtr>(schema, compiledSchema));
  if(ins.second)
    compiledSchema=ins.first->second=xmlSchemaNewValidCtxt(xmlSchemaParse(xmlSchemaNewParserCtxt(schema.c_str())));
  else
    compiledSchema=ins.first->second;

  int ret=xmlSchemaValidateDoc(compiledSchema, doc); // validate using compiled/cached schema

  xmlFreeDoc(doc);
  if(ret!=0) return ret;
  return 0;
}

// convert <xmlMatrix> and <xmlVector> elements to octave notation e.g [2;7;5]
int toOctave(TiXmlElement *&e) {
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlMatrix") {
    string mat="[";
    for(TiXmlElement* row=e->FirstChildElement(); row!=0; row=row->NextSiblingElement()) {
      for(TiXmlElement* ele=row->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        mat+=ele->GetText();
        if(ele->NextSiblingElement()) mat+=",";
      }
      if(row->NextSiblingElement()) mat+=";\n";
    }
    mat+="]";
    TiXmlText text(mat);
    e->Parent()->InsertEndChild(text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlVector") {
    string vec="[";
    for(TiXmlElement* ele=e->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
      vec+=ele->GetText();
      if(ele->NextSiblingElement()) vec+=";";
    }
    vec+="]";
    TiXmlText text(vec);
    e->Parent()->InsertEndChild(text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    if(toOctave(c)!=0) return 1;
    if(c==0) break; // break if c was removed by toOctave at the line below
    c=c->NextSiblingElement();
  }

  return 0;
}

int embed(TiXmlElement *&e, const string &nslocation, map<string,string> &nsprefix, map<string,string> &units, ostream &dependencies) {
  try {
    if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
      octEval->octavePushParams();
      // check if only href OR child element (This is not checked by the schema)
      TiXmlElement *l=0, *dummy;
      for(dummy=e->FirstChildElement(); dummy!=0; l=dummy, dummy=dummy->NextSiblingElement());
      if((e->Attribute("href") && l && l->ValueStr()!=MBXMLUTILSPVNS"localParameter") ||
         (e->Attribute("href")==0 && (l==0 || l->ValueStr()==MBXMLUTILSPVNS"localParameter"))) {
        TiXml_location(e, "", ": Only the href attribute OR a child element (expect pv:localParameter) is allowed in embed!");
        return 1;
      }
      // check if attribute count AND counterName or none of both
      if((e->Attribute("count")==0 && e->Attribute("counterName")!=0) ||
         (e->Attribute("count")!=0 && e->Attribute("counterName")==0)) {
        TiXml_location(e, "", ": Only both, the count and counterName attribute must be given or none of both!");
        return 1;
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
        octave_value v=symbol_table::varval("ret");
        octEval->checkType(v, MBXMLUtils::OctaveEvaluator::ScalarType);
        count=int(round(v.double_value()));
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
          return 1;
        }
        cout<<"Read "<<file<<endl;
        TiXmlDocument *enewdoc=new TiXmlDocument;
        enewdoc->LoadFile(file.c_str()); TiXml_PostLoadFile(enewdoc);
        enew=enewdoc->FirstChildElement();
        map<string,string> dummy;
        incorporateNamespace(enew, nsprefix, dummy, &dependencies);
        // convert embeded file to octave notation
        cout<<"Process xml[Matrix|Vector] elements in "<<file<<endl;
        if(toOctave(enew)!=0) {
          TiXml_location(e, "  included by: ", "");
          return 1;
        }
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
          return 1;
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
            return 1;
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
        
        octave_value o((double)i);
        octEval->octaveAddParam(counterName, o);
        octEval->octaveEvalRet(onlyif, e);
        octave_value v=symbol_table::varval("ret");
        octEval->checkType(v, MBXMLUtils::OctaveEvaluator::ScalarType);
        if(round(v.double_value())==1) {
          cout<<"Embed "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
          if(i==1) e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
          else e=(TiXmlElement*)(e->Parent()->InsertAfterChild(e, *enew));
  
          // include a processing instruction with the count number
          TiXmlUnknown countNr;
          countNr.SetValue("?EmbedCountNr "+TiXml_itoa(i)+"?");
          e->InsertAfterChild(e->FirstChild(), countNr);
      
          // apply embed to new element
          if(embed(e, nslocation, nsprefix, units, dependencies)!=0) return 1;
        }
        else
          cout<<"Skip embeding "<<(file==""?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      if(enewdoc)
        delete enewdoc;
      else
        delete enew;
      octEval->octavePopParams();
      return 0;
    }
    else {
      // THIS IS A WORKAROUND! Actually not all Text-Elements should be converted but only the Text-Elements
      // of XML elementx of a type devived from pv:scalar, pv:vector, pv:matrix and pv:string. But for that a
      // schema aware processor is needed!
      if(e->GetText()) {
        // eval text node
        octEval->octaveEvalRet(e->GetText(), e);
        // convert unit
        if(e->Attribute("unit") || e->Attribute("convertUnit")) {
          map<string, octave_value> savedCurrentParam;
          octEval->saveAndClearCurrentParam();
          octEval->octaveAddParam("value", symbol_table::varval("ret")); // add 'value=ret', since unit-conversion used 'value'
          if(e->Attribute("unit")) { // convert with predefined unit
            octEval->octaveEvalRet(units[e->Attribute("unit")]);
            e->RemoveAttribute("unit");
          }
          if(e->Attribute("convertUnit")) { // convert with user defined unit
            octEval->octaveEvalRet(e->Attribute("convertUnit"));
            e->RemoveAttribute("convertUnit");
          }
          octEval->restoreCurrentParam();
        }
        // wrtie eval to xml
        e->FirstChild()->SetValue(octEval->octaveGetRet());
        e->FirstChild()->ToText()->SetCDATA(false);
      }
    
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
      if(embed(c, nslocation, nsprefix, units, dependencies)!=0) return 1;
      c=c->NextSiblingElement();
    }
  }
  catch(string str) {
    TiXml_location(e, "", ": "+str);
    return 1;
  }
 
  return 0;
}

string extractFileName(const string& dirfilename) {
  bfs::path p(dirfilename.c_str());
  return (--p.end())->generic_string();
}

#define PATHLENGTH 10240
int main(int argc, char *argv[]) {
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
  string OCTAVEDIR;
  char *env;
  SCHEMADIR=SCHEMADIR_DEFAULT; // default: from build configuration
  if(!bfs::exists((SCHEMADIR+"/http___openmbv_berlios_de_MBXMLUtils/parameter.xsd").c_str())) SCHEMADIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/schema"; // use rel path if build configuration dose not work
  if((env=getenv("MBXMLUTILSSCHEMADIR"))) SCHEMADIR=env; // overwrite with envvar if exist
  XMLDIR=XMLDIR_DEFAULT; // default: from build configuration
  if(!bfs::exists((XMLDIR+"/measurement.xml").c_str())) XMLDIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/xml"; // use rel path if build configuration dose not work
  if((env=getenv("MBXMLUTILSXMLDIR"))) XMLDIR=env; // overwrite with envvar if exist
  OCTAVEDIR=OCTAVEDIR_DEFAULT; // default: from build configuration
  if(!bfs::exists(OCTAVEDIR.c_str())) OCTAVEDIR=MBXMLUtils::getInstallPath()+"/share/mbxmlutils/octave"; // use rel path if build configuration dose not work
  if((env=getenv("MBXMLUTILSOCTAVEDIR"))) OCTAVEDIR=env; // overwrite with envvar if exist
  // OCTAVE_HOME
  string OCTAVE_HOME; // the string for putenv must has program life time
  if(getenv("OCTAVE_HOME")==NULL && bfs::exists((MBXMLUtils::getInstallPath()+"/share/octave").c_str())) {
    OCTAVE_HOME="OCTAVE_HOME="+MBXMLUtils::getInstallPath();
    putenv((char*)OCTAVE_HOME.c_str());
  }

  try {
    // initialize octave
    char **octave_argv=(char**)malloc(2*sizeof(char*));
    octave_argv[0]=(char*)malloc(6*sizeof(char*)); strcpy(octave_argv[0], "dummy");
    octave_argv[1]=(char*)malloc(3*sizeof(char*)); strcpy(octave_argv[1], "-q");
    octave_main(2, octave_argv, 1);
    int dummy;
    eval_string("warning('error','Octave:divide-by-zero');",true,dummy,0); // statement list
    eval_string("addpath('"+OCTAVEDIR+"');",true,dummy,0); // statement list
  
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
        eval_string("addpath('"+absmpath+"');",true,dummy,0); // statement list
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
      TiXmlDocument *paramxmldoc=NULL;
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
  
      // convert parameter file to octave notation
      if(paramxml!="none") {
        cout<<"Process xml[Matrix|Vector] elements in "<<paramxml<<endl;
        if(toOctave(paramxmlroot)!=0) throw(1);
      }
  
      // generate octave parameter string
      if(paramxml!="none") {
        cout<<"Generate octave parameter string from "<<paramxml<<endl;
        if(octEval->fillParam(paramxmlroot)!=0) throw(1);
      }
      if(paramxmldoc) delete paramxmldoc;
  
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
            cout<<"ERROR! Unit name "<<el2->Attribute("name")<<" is defined more than once."<<endl;
            throw(1);
          }
          units[el2->Attribute("name")]=el2->GetText();
        }
      delete mmdoc;
  
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
  
      // convert main file to octave notation
      cout<<"Process xml[Matrix|Vector] elements in "<<mainxml<<endl;
      if(toOctave(mainxmlroot)!=0) throw(1);
  
      // embed/validate/toOctave/unit/eval files
      if(embed(mainxmlroot, nslocation,nsprefix,units,dependencies)!=0) throw(1);
  
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
  catch(...) {
    do_octave_atexit();
    return 1;
  }
  do_octave_atexit();
  return 0;
}
