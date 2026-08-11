param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$File,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Extension
)

if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
    throw "File does not exist: $File"
}

if (-not $Extension.StartsWith(".")) {
    $Extension = ".$Extension"
}

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class ShellAssociation
{
    [DllImport("Shlwapi.dll", CharSet = CharSet.Unicode)]
    private static extern int AssocQueryString(
        uint flags,
        uint str,
        string pszAssoc,
        string pszExtra,
        [Out] char[] pszOut,
        ref uint pcchOut
    );

    private const uint ASSOCF_NONE = 0;
    private const uint ASSOCSTR_EXECUTABLE = 2;

    public static string GetExecutable(string extension)
    {
        uint size = 0;

        int hr = AssocQueryString(
            ASSOCF_NONE,
            ASSOCSTR_EXECUTABLE,
            extension,
            null,
            null,
            ref size
        );

        // ERROR_INSUFFICIENT_BUFFER
        if (hr != unchecked((int)0x8007007A) && hr != 0)
            Marshal.ThrowExceptionForHR(hr);

        char[] buffer = new char[size];

        hr = AssocQueryString(
            ASSOCF_NONE,
            ASSOCSTR_EXECUTABLE,
            extension,
            null,
            buffer,
            ref size
        );

        if (hr != 0)
            Marshal.ThrowExceptionForHR(hr);

        return new string(buffer).TrimEnd('\0');
    }
}
'@

$executable = [ShellAssociation]::GetExecutable($Extension)

if ([string]::IsNullOrWhiteSpace($executable)) {
    throw "No default application found for $Extension"
}

Write-Host "Extension : $Extension"
Write-Host "Executable: $executable"
Write-Host "File      : $File"

Start-Process -FilePath $executable -ArgumentList "`"$File`""
