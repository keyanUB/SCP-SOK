from .BaseModel import BaseModel 
import anthropic
import time
import traceback

class Claude(BaseModel):
        
    def __init__(self, api_key=None) -> None:
        self.model = anthropic.Anthropic(
            api_key=api_key,
        )
        self.ignore_lang_for_code_extraction = False

    def _convert_to_claude_messages(self, messages):

        claude_history = []
        
        for message in messages:
           
            if message["role"] == "user":
                claude_history.append(message)
            elif message["role"] == "assistant":
                claude_history.append(message)
            elif message["role"] == "system":
                claude_history.extend([{
                    "role": "user",
                    "content": message["content"]
                }, {
                    "content": "Understood",
                    "role": "assistant"
                }])

        return claude_history


    def _generate(self, messages):

        # print("********************************************\n\nMESSAGES")
        # for message in messages:
        #     print("ROLE:", message["role"])
        #     print("CONTENT:", message["content"])
        #     if (message["content"].strip()==""):
        #         return ""

        messages = self._convert_to_claude_messages(messages)
        try:
            completion = self.model.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=messages
            )

            # print("OUTPUT ")
            # print(completion.content[0].text)
        except Exception:
            # traceback.print_exc()
            time.sleep(90) # For errors due to rate limit
            return self._generate(messages)
            
        return completion.content[0].text

