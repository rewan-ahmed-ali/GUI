from tkinter import *
import tkinter as tk
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import cv2
import numpy as np
import os
# ============dataset========
def add_label_to_dataset(name, priority):
    # Process the label data here (e.g., store in a file or database)
    print("Name:", name)
    print("Priority:", priority)

def detect_facess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces
def generate_dataset():
        
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\haarcascade_frontalface_default.xml')
    # Get the name and priority from the input fields
    name = t1.get()
    priority = t2.get()
    # For each person, enter one numeric face id

    face_id = priority
    print("\n[INFO] Initializing face capture. Look at the camera and wait ...")
    # Call the function to store the label information in the dataset
    add_label_to_dataset(name, priority)
    # Initialize individual sampling face count   
    count = 0
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            #cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imwrite("C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\dataset\\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
        print(f"\rImages Captured: {count}", end="")  # Print the count of captured images

        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        if k == 27 or count >= 30:  # Exit when 'ESC' is pressed or 30 face samples are captured
            break
    # Do a bit of cleanup
    print("\n\n[INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
# ===============  
def dataset_button_clicked():
    generate_dataset()
# ===========train=========
def training():
    # Path for face image database
    path ="C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create();
    detector = cv2.CascadeClassifier("C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\haarcascade_frontalface_default.xml");
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\trainer\\trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
def train_button_clicked():
    training()
# ====================
def face_recognized():
    os.system('python3 /home/dexter1/Desktop/hello.py &  python3 /home/dexter1/Desktop/FaceRec.py  ')
# ====================
#gui
root = tk.Tk()
root.title("dexter robot")
root.geometry("900x500+200+50")
root.resizable(False, False)
root.configure(bg="light grey")
# Robuntu image
img = Image.open("C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\Loogo.png")
photo = ImageTk.PhotoImage(img)
img_label = Label(root,image=photo)
img_label.place(x=400, y=100)
# DU image
img2 = Image.open("C:\\Users\\rewan ahmed\\Desktop\\gui_after\\gui_after\\gui\\DU.png")
photo2 = ImageTk.PhotoImage(img2)
img2_label = Label(root,image=photo2)
img2_label.place(x=100, y=220)
#title
title = Label(root, text="Robuntu Team(FCAI| |DU)", font=("times new roman",30) ,bg="#0a064f",fg="white" )
title.place(x=0,y=0,relwidth=1)
# ===============================
l1 = Label(root, text="Name",  font = " 10",fg="white",bg="#326fa8")
l1.place(x=10,y=100)
t1 = Entry(root, bd=5)
t1.place(x=110,y=100)
# =============================
l2 = Label(root, text="Priority ", font = " 10",fg="white",bg="#326fa8")
l2.place(x=10,y=150)
t2 = Entry(root, bd=5)
t2.place(x=110,y=150)
# =======================
b1 = Button(root, text="Face Database", font="10",bg="#326fa8", fg="white",command=generate_dataset)
b1.place(x=50,y=450)
 
b2 = Button(root, text="Database trained", font="10",bg="#326fa8", fg="white",command=train_button_clicked)
b2.place(x=610,y=450)

b3 = Button(root, text="Face Recognize", font="10",bg="#326fa8", fg="white",command=face_recognized)
b3.place(x=330,y=450)

root.mainloop()