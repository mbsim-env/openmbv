#include <config.h>
#include <jni.h>
#if _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
# include <windows.h>

  #define STRSIZE 1024
  char oldDllDirectory[STRSIZE];
#endif

/* save current dll search path set the it to binDir
 * Note: an _ in a package name must be converted to _1 in a c-file. */
JNIEXPORT void JNICALL Java_de_mbsim_1env_openmbv_OpenMBVJNI_storeAndSetDLLSearchDirectory(JNIEnv *jenv, jclass jcls, jstring jbinDir) {
#if _WIN32
  /* get binDir as c string */
  const char *binDir=(*jenv)->GetStringUTFChars(jenv, jbinDir, 0);

  GetDllDirectory(STRSIZE, oldDllDirectory);
  SetDllDirectory(binDir);

  /* release binDir */
  (*jenv)->ReleaseStringUTFChars(jenv, jbinDir, binDir);
#endif
}

/* restore old dll search path
 * Note: an _ in a package name must be converted to _1 in a c-file. */
JNIEXPORT void JNICALL Java_de_mbsim_1env_openmbv_OpenMBVJNI_restoreDLLSearchDirectory(JNIEnv *jenv, jclass jcls) {
#if _WIN32
  SetDllDirectory(oldDllDirectory);
#endif
}
