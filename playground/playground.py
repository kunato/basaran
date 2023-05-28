import torch
from basaran.model import load_model
from utils.downloadv2 import download


if __name__ == "__main__":
    MODEL = "TheBloke/falcon-7b-instruct-GPTQ"
    if "falcon" in MODEL:
        from monkey_patch.falcon import inject_monkey_patch_falcon

        kwargs = {
            "device_map_auto": True,
            "trust_remote_code": True,
            "dtype": torch.float32,
            "inject_monkey_patch_fn": inject_monkey_patch_falcon(MODEL),
        }
        if "GPTQ" in MODEL:
            kwargs.update({"gptq": True})
            local_path_or_model = download(MODEL, "models")
        else:
            local_path_or_model = MODEL
    else:
        kwargs = {}
        local_path_or_model = MODEL
    print("load model: -> ", local_path_or_model)
    model = load_model(local_path_or_model, **kwargs)
    prompt = "hello"
    for choice in model(prompt=prompt, max_tokens=128, temperature=0.7, top_p=1.0):
        print(choice["text"])
