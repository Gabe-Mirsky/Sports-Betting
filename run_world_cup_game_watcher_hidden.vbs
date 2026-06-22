Option Explicit

Dim shell, fso, scriptDir, batPath, args, i
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(scriptDir, "run_world_cup_game_watcher.bat")
args = ""

For i = 0 To WScript.Arguments.Count - 1
    args = args & " " & QuoteArg(WScript.Arguments(i))
Next

shell.Run """" & batPath & """" & args, 0, False

Function QuoteArg(value)
    QuoteArg = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
