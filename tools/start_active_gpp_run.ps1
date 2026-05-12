param(
    [Parameter(Mandatory = $true)]
    [string]$Compiler,

    [Parameter(Mandatory = $true)]
    [string]$Source,

    [Parameter(Mandatory = $true)]
    [string]$Workspace
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $Workspace "tools\run_active_gpp.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    Write-Error "Runner not found: $runner"
    exit 67
}

Start-Process -FilePath "powershell.exe" -WorkingDirectory $Workspace -ArgumentList @(
    "-NoLogo",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $runner,
    "-Compiler",
    $Compiler,
    "-Source",
    $Source,
    "-Workspace",
    $Workspace,
    "-PauseOnExit"
)
