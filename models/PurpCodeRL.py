from .BaseModel import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PurpCodeRL(BaseModel):

    def __init__(self, api_key=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("purpcode/purpcode-14b-rl", device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("purpcode/purpcode-14b-rl", device_map="auto")
        self.ignore_lang_for_code_extraction = True
    
    def _format_messages(self, messages):

        formatted_prompts = []
        
        for message in messages:
            formatted_prompts.append(message["content"])
        
        return "\n".join(formatted_prompts)
    
    def _generate(self, messages, system_prompt=None):
        device = torch.cuda.current_device()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=1600)
        completion = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        print("OUTPUT ")
        print(completion)
        return completion
