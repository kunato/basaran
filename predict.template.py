import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from basaran.model import load_model

MODEL = "$MODEL"


class Predictor(BasePredictor):
    def setup(self):
        if 'falcon-40b' in MODEL:
            kwargs = {
                'device_map_auto': True,
                'trust_remote_code': True,
                'dtype': torch.bfloat16
            }
        else:
            kwargs = {}
        self.model = load_model(MODEL, **kwargs)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt"),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
    ) -> ConcatenateIterator[str]:
        print("start prediction", prompt)
        for choice in self.model(
            prompt=prompt, max_tokens=max_length, temperature=temperature, top_p=top_p
        ):
            yield choice['text']
