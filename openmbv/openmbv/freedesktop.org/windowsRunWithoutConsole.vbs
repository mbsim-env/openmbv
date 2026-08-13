' start the command given as command line arguments while hidding the cmd window
' use '"wscript.exe" ".../runWithoutConsole.vbs" ... as the registry key "shell/open/command" or from any other GUI application
args = ""
For i = 0 To WScript.Arguments.Count - 1
  args = args & " """ & WScript.Arguments(i) & """"
Next
args = Mid(args, 2)
CreateObject("Wscript.Shell").Run args, 0, True
