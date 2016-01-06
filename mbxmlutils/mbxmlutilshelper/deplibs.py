#!/usr/bin/python

from __future__ import print_function
import sys
import subprocess
import re
import os
import distutils.spawn
import platform
import glob

if sys.version_info[0]==2: # to unify python 2 and python 3
  def lru_cache(maxsize=128, typed=False): # no lru_cache for python2, just just a empty decorator
    def wrapper(func): return func
    return wrapper
else:
  from functools import lru_cache



# @lru_cache(maxsize=None) the cache is very importend for this function!!!
def getWindowsEnvPath(name):
  if name in getWindowsEnvPath.res:
    return getWindowsEnvPath.res[name]
  else:
    if platform.system()=="Windows":
      getWindowsEnvPath.res[name]=os.environ[name]
      return getWindowsEnvPath.res[name]
    if platform.system()=="Linux":
      value=subprocess.check_output(["wine", "cmd", "/c", "echo", "%"+name+"%"], stderr=open(os.devnull,"w")).decode('utf-8').rstrip('\r\n')
      ret=[]
      cmd=["winepath", "-u"]
      cmd.extend(value.split(';'))
      vwin=subprocess.check_output(cmd, stderr=open(os.devnull,"w")).decode('utf-8').splitlines()
      if name=="PATH" and "WINEPATH" in os.environ:
        vwin=os.environ["WINEPATH"].split(";")+vwin
      getWindowsEnvPath.res[name]=';'.join(vwin)
      return getWindowsEnvPath.res[name]
    raise RuntimeError('Unknown platform')
getWindowsEnvPath.res={}

def searchWindowsLibrary(libname, libdir):
  searchDir=[] # is search in order
  searchDir.append(libdir)
  searchDir.append(getWindowsEnvPath('WINSYSDIR'))
  searchDir.append(getWindowsEnvPath('WINDIR'))
  searchDir.append(os.getcwd())
  searchDir.extend(getWindowsEnvPath('PATH').split(';'))
  for d in searchDir:
    for f in glob.glob(d+'/*'):
      if os.path.basename(f.upper())==libname.upper():
        return f
  raise RuntimeError('Library '+libname+' not found')

@lru_cache(maxsize=None)
def getDependencies(filename):
  res=set()
  content=subprocess.check_output(["file", "-L", filename], stderr=open(os.devnull,"w")).decode('utf-8')
  if re.search('ELF [0-9]+-bit LSB executable', content)!=None or re.search('ELF [0-9]+-bit LSB shared object', content)!=None:
    if re.search('statically linked', content)!=None:
      return res # skip static executables
    for line in subprocess.check_output(["ldd", filename], stderr=open(os.devnull,"w")).decode('utf-8').splitlines():
      match=re.search("^\s*(.+)\s=>\snot found$", line)
      if match!=None:
        raise RuntimeError('Library '+match.expand("\\1")+' not found')
      match=re.search("^.*\s=>\s(.+)\s\(0x[0-9a-fA-F]+\)$", line)
      if match!=None and not os.path.realpath(match.expand("\\1")) in getDoNotAdd():
        res.add(match.expand("\\1"))
    return res
  elif re.search('PE32\+? executable', content)!=None:
    try:
      for line in subprocess.check_output(["objdump", "-p", filename], stderr=open(os.devnull,"w")).decode('utf-8').splitlines():
        match=re.search("^\s*DLL Name:\s(.+)$", line)
        if match!=None:
          absfile=searchWindowsLibrary(match.expand("\\1"), os.path.dirname(filename))
          if not os.path.realpath(absfile) in getDoNotAdd():
            res.add(absfile)
      return res
    except subprocess.CalledProcessError:
      return res
  raise RuntimeError(filename+' unknown executable format')

# @lru_cache(maxsize=None) the cache is very importend for this function!!!
def getDoNotAdd():
  if getDoNotAdd.retCache!=None:
    return getDoNotAdd.retCache

  getDoNotAdd.retCache=set()

  # for linux
  system=[
    ("equery", ["-C", "files"], ["glibc", "mesa",]), # portage
    ("rpm",    ["-ql"],         ["glibc", "mesa-libGL", "mesa-libEGL",]), # rpm
  ]
  for s in system:
    for p in s[2]:
      if distutils.spawn.find_executable(s[0]):
        try:
          for line in subprocess.check_output([s[0]]+s[1]+[p], stderr=open(os.devnull,"w")).decode('utf-8').splitlines():
            getDoNotAdd.retCache.add(os.path.realpath(line))
        except subprocess.CalledProcessError:
          print('WARNING: Cannot get files of system package '+p+' using '+s[0]+' command', file=sys.stderr)

  # for windows
  getDoNotAdd.retCache.update(glob.glob("/home/mbsim/.wine/drive_c/windows/system32/*"))

  return getDoNotAdd.retCache
getDoNotAdd.retCache=None

@lru_cache(maxsize=None)
def relDir(filename):
  content=subprocess.check_output(["file", "-L", filename], stderr=open(os.devnull,"w")).decode('utf-8')
  if re.search('ELF [0-9]+-bit LSB *executable', content)!=None or re.search('ELF [0-9]+-bit LSB *shared object', content)!=None:
    return "lib" # Linux
  elif re.search('PE32\+? executable', content)!=None:
    return "bin" # Windows
  else:
    raise RuntimeError(filename+' unknwon executable format')

@lru_cache(maxsize=None)
def depLibs(filename):
  ret=set()
  content=subprocess.check_output(["file", "-L", filename], stderr=open(os.devnull,"w")).decode('utf-8')
  if re.search('ELF [0-9]+-bit LSB executable', content)!=None or re.search('ELF [0-9]+-bit LSB shared object', content)!=None:
    # on Linux search none recursively using ldd
    ret=getDependencies(filename)
  elif re.search('PE32\+? executable', content)!=None:
    # on Windows search recursively using objdump
    def walkDependencies(filename, deps):
      if filename not in deps:
        res=getDependencies(filename)
        deps.add(filename)
        for d in res:
          walkDependencies(d, deps)
    walkDependencies(filename, ret)
    # remove the library itself
    ret.discard(filename)
  
  return ret



# when started as a program
if __name__=="__main__":
  deps=depLibs(sys.argv[1])
  reldir=relDir(sys.argv[1])
  print('<DependentShFiles>')
  for d in deps:
    print('  <file reldir="%s" orgdir="%s">%s</file>'%(reldir, os.path.dirname(d), os.path.basename(d)))
  print('</DependentShFiles>')
