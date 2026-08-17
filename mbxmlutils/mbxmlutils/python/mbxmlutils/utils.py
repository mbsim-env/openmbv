def getDefaultOpenExeForExt(ext):
  import platform
  if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes
    
    shlwapi = ctypes.WinDLL("shlwapi.dll")
    shlwapi.AssocQueryStringW.argtypes = [
      ctypes.wintypes.DWORD,   # ASSOCF flags
      ctypes.wintypes.DWORD,   # ASSOCSTR
      ctypes.wintypes.LPCWSTR, # pszAssoc
      ctypes.wintypes.LPCWSTR, # pszExtra
      ctypes.wintypes.LPWSTR,  # pszOut
      ctypes.POINTER(ctypes.wintypes.DWORD), # pcchOut
    ]
    shlwapi.AssocQueryStringW.restype = ctypes.c_long
    ASSOCF_NONE = 0
    ASSOCSTR_EXECUTABLE = 2
    
    size = ctypes.wintypes.DWORD(0)
    if shlwapi.AssocQueryStringW(ASSOCF_NONE, ASSOCSTR_EXECUTABLE, ext, None, None, ctypes.byref(size)) != 1:
      raise RuntimeError("AssocQueryStringW failed")
    buf = ctypes.create_unicode_buffer(size.value)
    if shlwapi.AssocQueryStringW(ASSOCF_NONE, ASSOCSTR_EXECUTABLE, ext, None, buf, ctypes.byref(size)) != 0:
      raise RuntimeError("AssocQueryStringW failed")
    if len(buf.value)>=2 and buf.value[0]=='"' and buf.value[-1]=='"':
      return buf.value[1:-1]
    return buf.value
  elif platform.system() == "Linux":
    import os
    if ext=="txt" or ext==".txt":
      EDITOR = os.environ.get("EDITOR", None)
      if EDITOR is not None:
        return EDITOR
      else:
        raise RuntimeError("On platform Linux getDefaultOpenExeForExt .txt only works if the EDITOR envvar is set")
    else:
      raise RuntimeError("On platform Linux getDefaultOpenExeForExt is only implemented for .txt")
  else:
    raise RuntimeError(f"Unknown platform {platform.system()}")



if __name__ == "__main__":
  import sys
  if len(sys.argv) < 2:
    raise RuntimeError("At least one argument must be present")
  if sys.argv[1] == "openFileWithTextEditor":
    import subprocess
    import platform
    args={}
    if platform.system() == "Windows":
      args["creationflags"]=subprocess.CREATE_NO_WINDOW
    for f in sys.argv[2:]:
      subprocess.Popen([getDefaultOpenExeForExt(".txt"), f], **args)
  else:
    raise RuntimeError(f"Unknown first argument {sys.argv[1]}")
