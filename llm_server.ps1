$MODEL = ".\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

$SERVER = "C:\Users\Owner\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

& $SERVER `
    -m $MODEL `
    --port 8080 `
    --n-gpu-layers 0