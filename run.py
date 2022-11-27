from project import *


def press_button(entry):
    fileName = entry.get()	
    process_video(fileName)


if __name__ == "__main__":
    buildDB_b()
    buildDB_D()
    buildDB()

    screen = Tk(className='Plate Detector')
    screen.geometry('500x500')
    label = Label(screen, text='File Name')
    label.place(x=80, y=130)
    entry = Entry(screen)
    entry.place(x=240,y=130)

    button = Button(screen, text='OK', width=20, bg='brown', fg='white', command= lambda: press_button(entry))
    button.place(x=180, y=380)
    screen.mainloop()
