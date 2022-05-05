from abc import ABC, abstractclassmethod

class Layer:

    def __init__(self):
        pass
    
    @abstractclassmethod
    def forward(self,input):
        pass
