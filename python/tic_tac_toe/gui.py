from core import Core
from tkinter import Tk, Button
from tkinter.font import Font


class GUI:

    def __init__(self):
        self.app = Tk()
        self.app.title('TicTacToe')
        self.app.resizable(width=False, height=False)
        self.core = Core()
        self.font = Font(family="Helvetica", size=32)
        self.buttons = {}
        for x, y in self.core.fields:
            def handler(x=x, y=y): return self.move(x, y)
            button = Button(self.app, command=handler,
                            font=self.font, width=2, height=1)
            button.grid(row=y, column=x)
            self.buttons[x, y] = button

        def handler(): return self.reset()
        button = Button(self.app, text='reset', command=handler)
        button.grid(row=self.core.size+1, column=0,
                    columnspan=self.core.size, sticky="WE")
        self.update()

    def reset(self):
        self.core = Core()
        self.update()

    def move(self, x, y):
        self.app.config(cursor="watch")
        self.app.update()
        self.core = self.core.move(x, y)
        self.update()
        move = self.core.best()
        if move:
            self.core = self.core.move(*move)
            self.update()
        self.app.config(cursor="")

    def update(self):
        for (x, y) in self.core.fields:
            text = self.core.fields[x, y]
            self.buttons[x, y]['text'] = text
            self.buttons[x, y]['disabledforeground'] = 'black'
            if text == self.core.empty:
                self.buttons[x, y]['state'] = 'normal'
            else:
                self.buttons[x, y]['state'] = 'disabled'
        winning = self.core.won()
        if winning:
            for x, y in winning:
                self.buttons[x, y]['disabledforeground'] = 'red'
            for x, y in self.buttons:
                self.buttons[x, y]['state'] = 'disabled'
        for (x, y) in self.core.fields:
            self.buttons[x, y].update()

    def mainloop(self):
        self.app.mainloop()


if __name__ == '__main__':
    GUI().mainloop()
