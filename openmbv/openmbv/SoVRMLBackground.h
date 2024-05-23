/**************************************************************************\
 * Copyright (c) Kongsberg Oil & Gas Technologies AS
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**************************************************************************/

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// This is a copy of the same file from Coin3D https://github.com/coin3d/coin
// and adds the pull request https://github.com/coin3d/coin/pull/517.
// The class name is renamed from SoVRMLBackground to SoVRMLBackground2.
// In initClass instances of SoVRMLBackground are overwritten by instances of
// SoVRMLBackground2 to apply the above pull request to Background instances
// in wrl files.
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifndef OPENMBV_SOVRMLBACKGROUND_H
#define OPENMBV_SOVRMLBACKGROUND_H

#if __GNUC__ >= 12
  // gcc >= 12 release build triggers a false positive on this code
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#include <Inventor/nodes/SoSubNode.h>
#if __GNUC__ >= 12
  #pragma GCC diagnostic pop
#endif
#include <Inventor/nodes/SoNode.h>
#include <Inventor/fields/SoMFColor.h>
#include <Inventor/fields/SoMFFloat.h>
#include <Inventor/fields/SoMFString.h>
#include <Inventor/fields/SoSFBool.h>

class SoVRMLBackground2P;

#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

class DLL_PUBLIC SoVRMLBackground2 : public SoNode
{
  typedef SoNode inherited;
  SO_NODE_HEADER(SoVRMLBackground2);

public:
  static void initClass(void);
  SoVRMLBackground2(void);

  SoMFColor groundColor;
  SoMFFloat groundAngle;
  SoMFColor skyColor;
  SoMFFloat skyAngle;
  SoMFString backUrl;
  SoMFString bottomUrl;
  SoMFString frontUrl;
  SoMFString leftUrl;
  SoMFString rightUrl;
  SoMFString topUrl;

  virtual void GLRender( SoGLRenderAction * action );

protected:
  virtual SbBool readInstance(SoInput * in, unsigned short flags);
  virtual ~SoVRMLBackground2();

  SoSFBool set_bind; // eventIn
  SoSFBool isBound;  // eventOut

private:
  SoVRMLBackground2P * pimpl;

};

#endif // ! OPENMBV_SOVRMLBACKGROUND_H
