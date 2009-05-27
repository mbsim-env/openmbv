' This file starts OpenMBV by clicking on a object in MS Powerpoint.
' OpenMBV is started with all GUI widget closed and no window decoration.
' The viewing area is scaled and moved, such that is overlays the clicked
' object.
'
' 1. Import this Visual-Basic-Script in MS Powerpoint
' 2. Change the path to 'openmbv.exe'. See the line after 'EDIT!!! OpenMBV-PATH'
' 3. Change or duplicate the 'OpenMBV_path1' subroutine for each instance of OpenMBV
'    and change the OpenMBV parameters. See the line after 'EDIT!!! *.ombv.xml PATH'
' 4. Set the Powerpoint 'User Action' of the object on click to the macro 'OpenMBV_path1'

Type RECT
  x1 As Long
  y1 As Long
  x2 As Long
  y2 As Long
End Type

Declare Function GetDesktopWindow Lib "User32" () As Long
Declare Function GetWindowRect Lib "User32" (ByVal hWnd As Long, rectangle As RECT) As Long
          
Sub OpenMBV(para As String, objShape As Shape)
  ' display resulution
  Dim R As RECT
  dummy = GetWindowRect(GetDesktopWindow(), R)
  RESX = R.x2 - R.x1
  RESY = R.y2 - R.y1
  ' object resolution
  w = objShape.Width
  h = objShape.Height
  x = objShape.Left
  y = objShape.Top
  wp = w / 720 * RESX
  hp = h / 540 * RESY
  xp = x / 720 * RESX
  yp = y / 540 * RESY
  ' command
  ' EDIT!!! OpenMBV-PATH
  cmd = "h:\openmbv.exe --closeall --nodecoration --geometry " & _
    Round(wp) & "x" & Round(hp) & "+" & _
    Round(xp) & "+" & Round(yp) & " " & para
  Shell (cmd)
End Sub

Sub OpenMBV_path1(objShape As Shape)
  ' The first argument of the OpenMBV call is passed as a parameter list to openmbv.exe
  ' EDIT!!! *.ombv.xml PATH
  Call OpenMBV(".", objShape)
End Sub
