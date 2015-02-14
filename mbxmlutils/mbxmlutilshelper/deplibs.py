#!/usr/bin/python

from __future__ import print_function
import sys
import subprocess
import re
import os
import distutils.spawn
import platform
import glob

def getWindowsEnvPath(name):
  if platform.system()=="Windows":
    return os.environ[name]
  if platform.system()=="Linux":
    value=subprocess.check_output(["wine", "cmd", "/c", "echo", "%"+name+"%"], stderr=open(os.devnull,"w")).decode('utf-8').rstrip('\r\n')
    ret=[]
    for v in value.split(';'):
      vwin=subprocess.check_output(["winepath", "-u", v], stderr=open(os.devnull,"w")).decode('utf-8').rstrip('\r\n')
      ret.append(vwin)
    return ';'.join(ret)
  raise RuntimeError('Unknown platform')
def searchWindowsLibrary(libname, libdir):
  if searchWindowsLibrary.searchDir==None:
    searchWindowsLibrary.searchDir=[] # is search in order
    searchWindowsLibrary.searchDir.append(libdir)
    searchWindowsLibrary.searchDir.append(getWindowsEnvPath('WINSYSDIR'))
    searchWindowsLibrary.searchDir.append(getWindowsEnvPath('WINDIR'))
    searchWindowsLibrary.searchDir.append(os.getcwd())
    searchWindowsLibrary.searchDir.extend(getWindowsEnvPath('PATH').split(';'))
  for d in searchWindowsLibrary.searchDir:
    for f in glob.glob(d+'/*'):
      if os.path.basename(f.upper())==libname.upper():
        return f
  raise RuntimeError('Library '+libname+' not found')
searchWindowsLibrary.searchDir=None

def getDependencies(filename):
  try:
    return getDependencies.result[filename]
  except KeyError:
    pass
  res=[set(), True]
  getDependencies.result[filename]=res
  try:
    content=subprocess.check_output(["objdump", "-a", filename], stderr=open(os.devnull,"w")).decode('utf-8')
  except subprocess.CalledProcessError:
    print('WARNING: '+filename+' is not a real library. Skipping.', file=sys.stderr)
    res[1]=False
    return res
  if content.find('file format elf')>=0: # Linux
    for line in subprocess.check_output(["ldd", filename], stderr=open(os.devnull,"w")).decode('utf-8').split('\n'):
      match=re.search("^.*\s=>\s(.*)\s\(0x[0-9a-fA-F]+\)$", line)
      if match!=None:
        if match.expand("\\1")!="":
          res[0].add(match.expand("\\1"))
    res[1]=True
    return res
  elif content.find('file format pei')>=0: # Windows
    for line in subprocess.check_output(["objdump", "-p", filename], stderr=open(os.devnull,"w")).decode('utf-8').split('\n'):
      match=re.search("^\s*DLL Name:\s(.*)$", line)
      if match!=None:
        res[0].add(searchWindowsLibrary(match.expand("\\1"), os.path.dirname(filename)))
    res[1]=True
    return res
  else:
    raise RuntimeError('Unknown extension')
getDependencies.result=dict()

def getDoNotAdd():
  notAdd=set()
  system=[("equery", ["files", "glibc"]), # Gentoo
          ("dpkg",   ["-L", "glibc"]),    # debian
          ("rpm",    ["-ql", "glibc"])]   # rpm
  for s in system:
    if distutils.spawn.find_executable(s[0]):
      for line in subprocess.check_output([s[0]]+s[1], stderr=open(os.devnull,"w")).decode('utf-8').split('\n'):
        notAdd.add(line)
      return notAdd
  return notAdd

def walkDependencies(filename, deps):
  if filename not in deps:
    res=getDependencies(filename)
    if res[1]:
      deps.add(filename)
    for d in res[0]:
      walkDependencies(d, deps)

deps=set()
walkDependencies(sys.argv[1], deps)
# remove the library itself
deps.remove(sys.argv[1])

for n in getDoNotAdd():
  if n in deps:
    deps.remove(n)

if os.path.splitext(sys.argv[1])[1]==".dll":
  reldir="bin"
else:
  reldir="lib"
print('<DependentShFiles>')
for d in deps:
  print('  <file reldir="%s" orgdir="%s">%s</file>'%(reldir, os.path.dirname(d), os.path.basename(d)))
print('</DependentShFiles>')
