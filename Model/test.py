

class Base:
    def __init__(self):
        self.yum = 0

    def uh(self):
        print("Hello")


class Thingy(Base):
    def __init__(self, hat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hat = hat


a = Thingy("green")
a.uh()
print(a.hat)