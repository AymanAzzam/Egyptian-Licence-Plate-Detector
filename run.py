from project import *
from database import build_databases


def press_button(entry, hamzaNo2taDB, barNesrDB, charactersDB):
    fileName = entry.get()	
    process_video(fileName, hamzaNo2taDB, barNesrDB, charactersDB)


if __name__ == "__main__":
    hamzaNo2taDB, barNesrDB, charactersDB = build_databases()
    
    screen = Tk(className='Plate Detector')
    screen.geometry('500x500')
    label = Label(screen, text='File Name')
    label.place(x=80, y=130)
    entry = Entry(screen)
    entry.place(x=240,y=130)

    button = Button(screen, text='OK', width=20, bg='brown', fg='white', command= lambda: press_button(entry, hamzaNo2taDB, barNesrDB, charactersDB))
    button.place(x=180, y=380)
    screen.mainloop()
