#include <jni.h>
#if _WIN32
# include <windows.h>

  #define STRSIZE 1024
  char oldDllDirectory[STRSIZE];
#endif

/* save current dll search path set the it to binDir */
JNIEXPORT void JNICALL Java_de_berlios_openmbv_OpenMBV_OpenMBVJNI_storeAndSetDLLSearchDirectory(JNIEnv *jenv, jclass jcls, jstring jbinDir) {
#if _WIN32
  /* get binDir as c string */
  const char *binDir=(*jenv)->GetStringUTFChars(jenv, jbinDir, 0);

  GetDllDirectory(STRSIZE, oldDllDirectory);
  SetDllDirectory(binDir);

  /* release binDir */
  (*jenv)->ReleaseStringUTFChars(jenv, jbinDir, binDir);
#endif
}

/* restore old dll search path */
JNIEXPORT void JNICALL Java_de_berlios_openmbv_OpenMBV_OpenMBVJNI_restoreDLLSearchDirectory(JNIEnv *jenv, jclass jcls) {
#if _WIN32
  SetDllDirectory(oldDllDirectory);
#endif
}
