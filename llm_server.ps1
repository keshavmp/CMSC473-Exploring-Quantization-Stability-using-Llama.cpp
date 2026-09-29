# Serve a GGUF from this repo's models\ folder. Paths anchor to this script.
param(
    [string]$Model = "models\block1-Q8.gguf",
    [int]$Port = 8080,
    [string]$Server = "llama-server.exe"
)

$ErrorActionPreference = "Stop"
$Model = if ([IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $PSScriptRoot $Model }

if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Server)) { $Server = (Get-Command $Server -ErrorAction SilentlyContinue).Source }
if (-not $Server) { throw "llama-server not found. Pass -Server <path>." }

& $Server -m $Model --port $Port --n-gpu-layers 0
