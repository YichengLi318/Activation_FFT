import os
from huggingface_hub import snapshot_download


def main():
    repo_id = 'Qwen/Qwen3-1.7B'
    local_dir = os.path.join('model', 'Qwen3-1.7B')
    os.makedirs(local_dir, exist_ok=True)

    token = os.environ.get('HF_TOKEN', None)
    print(f"Downloading '{repo_id}' to '{local_dir}'...")
    if token:
        print("Using HF_TOKEN from environment.")
    else:
        print("No HF_TOKEN found. If the repo is gated, ensure you're logged in or set HF_TOKEN (Qwen3-1.7B is public).")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    print("Download complete.")


if __name__ == '__main__':
    main()