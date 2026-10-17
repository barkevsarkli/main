# Release build on Windows (MSVC or clang-cl via the Visual Studio generator, or Ninja).
# Binaries land in version6\bin\.
#   .\build.ps1            configure + build
#   .\build.ps1 -Test      ... then run the C++ tests (ctest)
param([switch]$Test)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cmake -S . -B build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build build --config Release --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Test) {
    Push-Location build
    ctest -C Release --output-on-failure
    $rc = $LASTEXITCODE
    Pop-Location
    exit $rc
}
