import json

class Config:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.loaded = False
        self.data = {}
    def load(self):
        if( not self.loaded):
            self.Reload()
        self.loaded = True
    def Reload(self):
        with open(self.config_path, "r") as fp:
            self.data = json.load(fp)
    def Save(self):
        with open(self.config_path, "w") as fp:
            json.dump(self.data, fp)
    def __getitem__(self, item):
        if(not self.loaded):
            raise Exception("Load config first")
        try:
            return self.data[item]
        except:
            raise Exception("{} is not found!".format(item))
    def __setitem__(self, item, value):
        self.data[item] = value

all_config = Config("./config.json")
