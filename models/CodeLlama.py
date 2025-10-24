from .BaseModel import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeLlama(BaseModel):

    def __init__(self, api_key=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", device_map="auto")
        self.ignore_lang_for_code_extraction = False

    def _format_messages(self, messages):
        
        messages = [message for message in messages if message["role"] != "system"]
        system_prompt = [message for message in messages if message["role"] == "system"]
        if not len(system_prompt):
            system_prompt = "You are a helpful AI assistant"
        else:
            system_prompt = system_prompt[0]
        
        formatted_prompt = "<s>[INST] <<SYS>> " + system_prompt + " <</SYS>> \n"
        
        for idx in range(len(messages)):
            if (idx==0): #first user prompt
                formatted_prompt +=  messages[idx]["content"] + " [/INST]"
            elif (idx==len(messages)-1): #last user prompt
                formatted_prompt += f" </s><s>[INST]" + messages[idx]["content"] + " [/INST]"
            elif (idx%2==0): #intermediate user prompts
                formatted_prompt += f"[INST]" + messages[idx]["content"] + " [/INST]"
            elif (idx%2==1): #intermediate agent outputs
                formatted_prompt += messages[idx]["content"]
        
        return formatted_prompt + "\n"
    
    def _generate(self, messages, system_prompt=None):
        
        messages = self._format_messages(messages)
        print("********************************************\n\nMESSAGES")
        print(messages)
        device = torch.cuda.current_device()
        inputs = self.tokenizer(messages, return_tensors="pt").to(device)
        generated_tokens = self.model.generate(inputs.input_ids, max_new_tokens=1600)
        completion = self.tokenizer.batch_decode(generated_tokens)[0]
        # Get last answer
        completion = completion.split('[/INST]')[-1]
        print("OUTPUT ")
        print(completion)
        return completion
    