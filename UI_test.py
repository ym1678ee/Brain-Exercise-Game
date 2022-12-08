from tkinter import *
import turtle as t


window_1 = Tk()
window_1.title("Brain Exercise")
window_1.geometry('800x600')

intro_1 = Label(window_1, text = "좌뇌 우뇌 운동", font=('Arial', 30))
intro_1.place(x=300, y = 70)

intro_2 = Label(window_1, text="하고싶은 게임을 선택하세요", font=('Arial', 15))
intro_2.place(x=295, y=150)

def B():
    import Bear
    Bear.BearLeg()
    
def H():
    import Hand
    Hand.Hand()
    
    
    
Button_1 = Button(window_1, overrelief="solid", width=15, height=3, text="Bear", highlightcolor="yellow", command = B)
Button_1.place(x=120, y=300)
Button_2 = Button(window_1, overrelief="solid", width=15, height=3, text="Hand", highlightcolor="yellow", command = H)
Button_2.place(x=300, y=300)
Button_3 = Button(window_1, overrelief="solid", width=20, height=3, text="How to Play", highlightcolor="yellow")
Button_3.place(x=500, y=300)
window_1.mainloop()