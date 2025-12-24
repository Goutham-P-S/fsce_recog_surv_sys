import os
import requests
import sys

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    sys.stdout.write(f"\rProgress: {percent:.1f}%")
                    sys.stdout.flush()
        
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def main():
    MODELS = {
        "deepfillv2": {
            "url": "https://huggingface.co/ford442/deepfillv2-inpainting/resolve/main/onnx/deepfillv2.onnx",
            "path": "models/inpainting/deepfillv2.onnx"
        },
        "hifill": {
            "url": "https://huggingface.co/ford442/deepfillv2-inpainting/resolve/main/onnx/hifill.onnx",
            "path": "models/inpainting/hifill.onnx"
        },
        "lama": {
            "url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            "path": "models/inpainting/lama.onnx"
        }
    }
    
    print("--- Model Downloader ---")
    for name, info in MODELS.items():
        if os.path.exists(info['path']):
            print(f"Model '{name}' already exists at {info['path']}")
            continue
            
        success = download_file(info['url'], info['path'])
        if not success:
            print(f"Failed to download {name}.")
            sys.exit(1)
            
    print("All models ready.")

if __name__ == "__main__":
    main()
