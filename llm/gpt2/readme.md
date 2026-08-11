## download model

url: https://huggingface.co/openai-community/gpt2
weights for gpt2 can be downloaded from the following commands:
```bash
mkdir -p weights && cd weights
curl -L -o vocab.bpe    https://openaipublic.blob.core.windows.net/gpt-2/models/117M/vocab.bpe
curl -L -o encoder.json https://openaipublic.blob.core.windows.net/gpt-2/models/117M/encoder.json
## download model weights
curl -L -o config.json "https://huggingface.co/openai-community/gpt2/resolve/main/config.json?download=true"
curl -L -o model.safetensors "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors?download=true"
cd ..
```
> If failed to download the model files, you can access the above urls and download them manually.