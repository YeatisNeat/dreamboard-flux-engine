# predict.py
# Clean slate version, following official documentation exactly.
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading google/flan-t5-base model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained(
            'google/flan-t5-base', 
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        print("Model loaded successfully.")

    def predict(
        self,
        prompt: str = Input(description="Text prompt to send to the model.")
    ) -> str:
        """Run a single prediction on the model"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=100)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
