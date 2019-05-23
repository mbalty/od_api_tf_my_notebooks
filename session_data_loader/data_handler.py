import json


class JsonHandler:
    def __init__(self, content):
        self.jsonContent = content

    def dump(self, path):
        with open(path, "W") as f:
            json.dump(self.jsonContent, f)

    @staticmethod
    def load(self, path, type=JsonHandler):
        with open(path, "r") as f:
            return type(json.load(f))


class SessionHandler(JsonHandler):
    @property
    def id(self):
        return self.jsonContent["sessionId"]

    
    @property
    def customerId(self):
        return self.jsonContent["customerId"]

    @property
    def start(self):
        return self.jsonContent["start"]


    @property
    def end(self):
        return self.jsonContent["end"]


class BasketChangeHandler:
    @property
    def id(self):
        return self.jsonContent["sessionId"]

    @property
    def 