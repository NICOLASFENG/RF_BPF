param(
    [Parameter(Mandatory = $true)]
    [string]$Compiler,

    [Parameter(Mandatory = $true)]
    [string]$Source,

    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [switch]$PauseOnExit
)

$ErrorActionPreference = "Stop"

function Finish-Run {
    param([int]$Code)

    if ($PauseOnExit) {
        Write-Host ""
        Read-Host "Press Enter to close"
    }

    exit $Code
}

if (-not (Test-Path -LiteralPath $Compiler)) {
    Write-Host "Compiler not found: $Compiler"
    Finish-Run 127
}

if (-not (Test-Path -LiteralPath $Source)) {
    Write-Host "Source file not found: $Source"
    Finish-Run 66
}

$buildDir = Join-Path $Workspace "build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$baseName = [System.IO.Path]::GetFileNameWithoutExtension($Source)
$output = Join-Path $buildDir "$baseName.exe"

$compileArgs = @(
    "-fdiagnostics-color=always",
    "-std=c++17",
    "-g3",
    "-O0",
    "-Wall",
    "-Wextra",
    "-pedantic",
    "-D_WIN32_WINNT=0x0601",
    "-finput-charset=UTF-8",
    "-fexec-charset=UTF-8",
    $Source,
    "-o",
    $output
)

Write-Host "g++ $Source -> $output"
& $Compiler @compileArgs
if ($LASTEXITCODE -ne 0) {
    Finish-Run $LASTEXITCODE
}

Write-Host ""
Write-Host "Running $output"
& $output
$runExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Exit code: $runExitCode"
Finish-Run $runExitCode
