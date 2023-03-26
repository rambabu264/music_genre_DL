import functions
import MFCC
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame


dict_genres = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Disco': 3,
               'Hip-Hop': 4, 'Jazz': 5, 'Metal': 6, 'Pop':  7,
               'Reggae': 8, 'Rock': 9
               }

reverse_map = {v: k for k, v in dict_genres.items()}


def choose_file():

    file_path = tk.filedialog.askopenfilename()
    return file_path


def make_predictions(model, file_path):
    feature_extractor = MFCC.Feature_Extractor(file_path)
    mfcc = feature_extractor.MFCCS()

    X = np.asarray(mfcc)
    X= X[:997]

    X = X.reshape(1,X.shape[0],X.shape[1],1)
    print(X.shape)

    prediction = model.predict(X)
    print(reverse_map[np.argmax(prediction[0])])

    graph_plot(prediction[0], list(dict_genres.keys()), file_path)


def graph_plot(data, labels, path):
    print(path)

    # create a GUI window
    window = tk.Tk()
    window.geometry('600x400')
    window.title("Music Genre Prediction")

    txt = "The style of this song is reminiscent of "+ str(reverse_map[np.argmax(data)]) + " !!"

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(data)), data, align='center', color='b')

    # Set the axis labels and title
    ax.set_xlabel('Music Genre')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Music Genres')

    # Set the tick labels on the x-axis
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add a grid and adjust the layout
    ax.grid(False)
    fig.tight_layout()

    # Show the plot
    plt.show()

    pygame.init()

    def play():
        pygame.mixer.music.load(path)  # Loading File Into Mixer
        pygame.mixer.music.play()  # Playing It In The Whole Device

    play_button = Button(window, text="Play Audio", command=play, font=('Times', 24))
    play_button.pack(pady=12)

    # create a message widget
    message = tk.Text(window, height=2, font=('Times', 24))
    message.insert(tk.END, txt)
    message.pack(pady=12)

    # create a canvas widget to display the figure
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # run the tkinter event loop
    window.mainloop()


if __name__ == "__main__":

    path = '../Data/genres_original'
    checkpoint_path = "saved_model/music_genre_model.h5"

    # model = functions.build_model(path)
    model = functions.load_model(checkpoint_path)

    # create a GUI window
    root = tk.Tk()

    button = tk.Button(root, text="Choose File", command=choose_file)
    button.pack()

    file_path = choose_file()
    print("Selected file: ", file_path)

    make_predictions(model, file_path)



