from .BaseModel import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Phi2Safecoder(BaseModel):

    def __init__(self, api_key=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("models/trained/phi-2-safecoder/checkpoint-last", device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("models/trained/phi-2-safecoder/checkpoint-last", device_map="auto")
        self.ignore_lang_for_code_extraction = True
    
    def _format_messages(self, messages):

        formatted_prompts = ""
        
        for message in messages:
            if message["role"]=="system":
                continue
            formatted_prompts+= message["content"]
        
        
        return f"Instruct: {formatted_prompts.strip()}\n\nOutput:"
    
    def _generate(self, messages, system_prompt=None):
        
        count_bobs = 0
        for message in messages:
            if message["role"]=='Bob':
                count_bobs+=1
        messages = self._format_messages(messages)
        print("********************************************\n\nMESSAGES")
        print(messages)
        device = torch.cuda.current_device()
        inputs = self.tokenizer(messages, return_tensors="pt").to(device)
        generated_tokens = self.model.generate(inputs.input_ids, temperature=0.4, top_p=0.95, do_sample=True, max_new_tokens=1000)
        completion = self.tokenizer.batch_decode(generated_tokens)[0]
        # Get last answer
        print("OUTPUT")
        print(completion)
        # completion = completion.split("Bob:")[count_bobs]
        # if "Alice:" in completion:
        #     completion = completion.split("Alice:")[0]
        # print("EXTRACTED OUTPUT ")
        # print(completion)
        return completion
