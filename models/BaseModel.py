class BaseModel():

    def __init__(self, api_key=None) -> None:
        pass
   
    def _generate(self, messages, config):
        raise NotImplementedError()
