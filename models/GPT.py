import time
from .BaseModel import BaseModel 
from openai import OpenAI
import traceback

class OpenAIGPT(BaseModel):
        
    def __init__(self, api_key=None, version="gpt-4o-mini") -> None:
        self.iteration = 0
        self.model = OpenAI(
            api_key=api_key,
        )
        if '3' in version:
            self.version = 'gpt-3.5-turbo'
        else:
            self.version = 'gpt-4o-mini'
        self.ignore_lang_for_code_extraction = False

    def _generate(self, messages, system_prompt=None, trial=0):
        
        # print("********************************************\n\nMESSAGES")
        # for message in messages:
        #     print("VERSION:", self.version)
        #     print("ROLE:", message["role"])
        #     print("CONTENT:", message["content"])

        try:
            completion = self.model.chat.completions.create(
                model=self.version,
                messages=messages
            )
            
            # print("OUTPUT ")
            # print(completion.choices[0].message.content)
        except Exception:
            if (trial<3):
                print("Error")
                time.sleep(60)
                return self._generate(messages, system_prompt=None, trial=trial+1)
            else:
                completion = "ERROR"
        return completion.choices[0].message.content
