#!/usr/bin/env python3

import sys
import subprocess
import re
import os
import shutil
import platform
import glob
import functools
import time
import argparse



@functools.lru_cache(maxsize=None)
def convertPath(name):
  if platform.system()=="Linux":
    return name
  return subprocess.check_output(["cygpath", "-w", name], stderr=open(os.devnull,"w")).decode('utf-8').rstrip('\r\n')

@functools.lru_cache(maxsize=None)
def getWindowsEnvPath(name):
  if platform.system()=="Windows":
    return os.environ[name]
  if platform.system()=="Linux":
    value=subprocess.check_output(["wine", "cmd", "/c", "echo", "%"+name+"%"], stderr=open(os.devnull,"w")).decode('utf-8').rstrip('\r\n')
    cmd=["winepath", "-u"]
    cmd.extend(value.split(';'))
    vwin=subprocess.check_output(cmd, stderr=open(os.devnull,"w")).decode('utf-8').splitlines()
    if name=="PATH" and "WINEPATH" in os.environ:
      vwin=os.environ["WINEPATH"].split(";")+vwin
    return ';'.join(vwin)
  raise RuntimeError('Unknown platform')

def skipWindowsLibrary(libname):
  # skip some libname's
  if libname.startswith("api-ms-win-"): return True # windows system libs
  if libname=="???": return True # ???
  if libname.startswith("libfltk"): return True # a octave dep of a unused octave module
  if libname.startswith("libportaudio"): return True # a octave dep of a unused octave module
  return False

def searchWindowsLibrary(libname, libdir, filename):
  if skipWindowsLibrary(libname):
    return None

  searchDir=[] # is search in order
  searchDir.append(libdir)
  searchDir.append(getWindowsEnvPath('WINDIR')+"/system")
  searchDir.append(getWindowsEnvPath('WINDIR')+"/system32")
  searchDir.append(getWindowsEnvPath('WINDIR'))
  searchDir.append(os.getcwd())
  searchDir.extend(getWindowsEnvPath('PATH').split(';'))
  for d in searchDir:
    for f in glob.glob(d+'/*'):
      if os.path.basename(f.upper())==libname.upper():
        return f
  raise RuntimeError('Library '+libname+' not found (a dependency of '+filename+')')

@functools.lru_cache(maxsize=None)
def getDependencies(filename):
  res=set()
  content=fileL(filename)
  if re.search('ELF [0-9]+-bit LSB.*executable', content) is not None or re.search('ELF [0-9]+-bit LSB.*shared object', content) is not None:
    if re.search('statically linked', content) is not None:
      return res # skip static executables
    lddOutput=subprocess.check_output(["ldd", filename], stderr=open(os.devnull,"w")).decode('utf-8')
    for line in lddOutput.splitlines():
      match=re.search("^\s*(.+)\s=>\snot found$", line)
      if match is not None and not skipWindowsLibrary(match.expand("\\1")):
        raise RuntimeError('Library '+match.expand("\\1")+' not found (a dependency of '+filename+')')
      match=re.search("^\s*(.+)\s=>\s(.+)\s\(0x[0-9a-fA-F]+\)$", line)
      if match is not None and not skipFile(os.path.realpath(convertPath(match.expand("\\2")))) and not skipWindowsLibrary(match.expand("\\1")):
        res.add(convertPath(match.expand("\\2")))
    return res
  elif re.search('PE32\+? executable', content) is not None:
    try:
      for line in subprocess.check_output(["objdump", "-p", filename], stderr=open(os.devnull,"w")).decode('utf-8').splitlines():
        match=re.search("^\s*DLL Name:\s(.+)$", line)
        if match is not None:
          absfile=searchWindowsLibrary(match.expand("\\1"), os.path.dirname(filename), filename)
          if absfile is not None and not skipFile(os.path.realpath(absfile)):
            res.add(absfile)
      return res
    except subprocess.CalledProcessError:
      return res
  raise RuntimeError(filename+' unknown executable format')

@functools.lru_cache(maxsize=None)
def getDoNotAdd():
  notAdd=set()
  # for linux
  system=[
    ("rpm", ["-ql"], # rpm (CentOS7)
      ["glibc", "zlib", "libstdc++", "libgcc", # standard libs
       "mesa-dri-drivers", "mesa-libGLU", "mesa-libgbm", "mesa-libglapi", "mesa-libEGL", "mesa-libGL", # mesa
       "libdrm", "libglvnd", "libglvnd-glx", "libglvnd-opengl", # mesa
       "libX11", "libXext", "libxcb", "libXau", # standard X11
      ]),
  ]
  for s in system:
    for p in s[2]:
      if shutil.which(s[0]):
        try:
          for line in subprocess.check_output([s[0]]+s[1]+[p], stderr=open(os.devnull,"w")).decode('utf-8').splitlines():
            notAdd.add(os.path.realpath(line))
        except subprocess.CalledProcessError:
          print('WARNING: Cannot get files of system package '+p+' using '+s[0]+' command', file=sys.stderr)

  # for linux do not add the (specific to docker image mbsimenv/build)
  notAdd.update(glob.glob("/osupdate/local/lib64/libglapi.so*")) # mesa
  notAdd.update(glob.glob("/osupdate/local/lib64/libGLESv1_CM.so*")) # mesa
  notAdd.update(glob.glob("/osupdate/local/lib64/libGLESv2.so*")) # mesa
  notAdd.update(glob.glob("/osupdate/local/lib64/libGL.so*")) # mesa

  # for windows-wine do not add the Windows system dlls (= fake dlls on wine)
  notAdd.update(glob.glob("/usr/lib64/wine/fakedlls/*")) # read wine fake dlls
  notAdd.update(glob.glob("/usr/lib/wine/fakedlls/*")) # read wine fake dlls
  notAdd.update(glob.glob(os.environ['HOME']+"/.wine/drive_c/windows/system32/*")) # copy in HOME dir
  notAdd.update(glob.glob(os.environ['HOME']+"/.wine/drive_c/windows/syswow64/*")) # copy in HOME dir
  # for windows-msys2 do not add the Windows system dlls
  notAdd.update(glob.glob(os.environ.get("WINDIR", "")+"/system32/*"))
  notAdd.update(glob.glob(os.environ.get("WINDIR", "")+"/syswow64/*"))

  notAdd=set(map(lambda x: x.replace("\\", "/").upper(), notAdd))

  return notAdd

def skipFile(absfile):
  return absfile.replace("\\", "/").upper() in getDoNotAdd()

@functools.lru_cache(maxsize=None)
def relDir(filename):
  content=fileL(filename)
  if re.search('ELF [0-9]+-bit LSB.*executable', content) is not None or re.search('ELF [0-9]+-bit LSB.*shared object', content) is not None:
    return "lib" # Linux
  elif re.search('PE32\+? executable', content) is not None:
    return "bin" # Windows
  else:
    raise RuntimeError(filename+' unknwon executable format')

@functools.lru_cache(maxsize=None)
def fileL(filename):
  return subprocess.check_output(["file", "-L", filename], stderr=open(os.devnull,"w")).decode('utf-8')

@functools.lru_cache(maxsize=None)
def depLibs(filename):
  ret=set()
  content=fileL(filename)
  if re.search('ELF [0-9]+-bit LSB.*executable', content) is not None or re.search('ELF [0-9]+-bit LSB.*shared object', content) is not None:
    # on Linux search none recursively using ldd
    ret=getDependencies(filename)
  elif re.search('PE32\+? executable', content) is not None:
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
  argparser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description='''Create dependency xml file'''
  )
  argparser.add_argument("files", nargs="*", help="dll/so files to process")
  outArgs = argparser.add_mutually_exclusive_group(required=True)
  outArgs.add_argument("-o", type=str, help="output files to <o>/<basename of input-file>.deplibs")
  outArgs.add_argument("-b", action="store_true", help="output files to <dir-of input-file>/<basename of input-file>.deplibs")
  argparser.add_argument("--skip", type=str, default=[], action="append", help="basefile-name to skip")
  args = argparser.parse_args()

  if args.o is not None:
    os.makedirs(args.o, exist_ok=True)

  for file in args.files:
    if os.path.basename(file) in args.skip:
      print("deplibs.py: skipping: "+file)
      continue

    if args.o is not None:
      outFile = os.path.join(args.o, os.path.basename(file)+".deplibs")
    else:
      outFile = os.path.join(os.path.dirname(file), os.path.basename(file)+".deplibs")

    if os.path.exists(outFile) and os.path.getmtime(outFile) > os.path.getmtime(file):
      print("deplibs.py: already up to date: "+outFile)
      continue

    print("deplibs.py: Creating: "+outFile)
    deps=depLibs(file)
    reldir=relDir(file)
    with open(outFile, "wt") as f:
      print('<DependentShFiles>', file=f)
      for d in deps:
        print('  <file reldir="%s" orgdir="%s">%s</file>'%(reldir, os.path.dirname(d), os.path.basename(d)), file=f)
      print('</DependentShFiles>', file=f)
