' This file starts OpenMBV by clicking on a object in MS Powerpoint.
' OpenMBV is started with all GUI widget closed and no window decoration.
' The viewing area is scaled and moved, such that is overlays the clicked
' object.
'
' 1. Import this Visual-Basic-Script in MS Powerpoint
' 2. Change the path to 'openmbv.exe'. See the line after
'    'EDIT!!! OpenMBV-PATH'
' 3. Change or duplicate the 'OpenMBV_path1' subroutine for each instance of
'    OpenMBV and change the OpenMBV parameters. See the line after
'    'EDIT!!! *.ombvx PATH'
' 4. Set the Powerpoint 'User Action' of the object on click to the macro
'    'OpenMBV_path1'
' 5. Run/Load the Add-In file "AutoEvents.ppa" before the presentation

Type RECT
  x1 As Long
  y1 As Long
  x2 As Long
  y2 As Long
End Type

Declare Function GetDesktopWindow Lib "user32" () As Long
Declare Function GetWindowRect Lib "user32" (ByVal hWnd As Long, _
  rectangle As RECT) As Long

Dim handleW1 As Long

Declare Function FindWindowA Lib "user32" (ByVal lpClassName As String, _
  ByVal lpWindowName As String) As Long

Declare Function SetWindowPos Lib "user32" (ByVal handleW1 As Long, _
  ByVal handleW1InsertWhere As Long, ByVal w As Long, ByVal x As Long, _
  ByVal y As Long, ByVal z As Long, ByVal wFlags As Long) As Long

Const TOGGLE_HIDEWINDOW = &H80
Const TOGGLE_UNHIDEWINDOW = &H40

Function HideTaskbar()
  handleW1 = FindWindowA("Shell_traywnd", "")
  Call SetWindowPos(handleW1, 0, 0, 0, 0, 0, TOGGLE_HIDEWINDOW)
End Function

Function UnhideTaskbar()
  Call SetWindowPos(handleW1, 0, 0, 0, 0, 0, TOGGLE_UNHIDEWINDOW)
End Function

Sub Auto_ShowBegin()
  Call HideTaskbar
End Sub
Sub Auto_ShowEnd()
  Call UnhideTaskbar
End Sub




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
  ' EDIT!!! *.ombvx PATH
  Call OpenMBV(".", objShape)
End Sub
