from .BaseModel import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Llama2Chat(BaseModel):

    def __init__(self, api_key=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
        self.ignore_lang_for_code_extraction = True
    
    def _format_messages(self, messages):

        formatted_prompts = ["<|begin_of_text|>"]
        
        for message in messages:
            formatted_prompts.append("<|start_header_id|>"+message["role"]+"<|end_header_id|>"+message["content"]+"<|eot_id|>")
        
        return "\n".join(formatted_prompts)
    
    def _generate(self, messages, system_prompt=None):
        
        # messages = self._format_messages(messages)
        print("********************************************\n\nMESSAGES")
        device = torch.cuda.current_device()
        messages = self.tokenizer.apply_chat_template(messages, tokenize=False)
        print(messages)

        inputs = self.tokenizer(messages, return_tensors="pt").to(device)
        generated_tokens = self.model.generate(inputs.input_ids, max_new_tokens=1600)
        completion = self.tokenizer.batch_decode(generated_tokens)[0]
        # Get last answer
        completion = completion.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        print("OUTPUT ")
        print(completion)
        return completion
