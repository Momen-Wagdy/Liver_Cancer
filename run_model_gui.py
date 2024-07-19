import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import filedialog, Label, Button, Tk
from PIL import Image, ImageTk

# Load the model once globally
model = load_model('best_model.h5')

# Define the prediction function
@tf.function
def predict(image):
    predictions = model(image)
    return 'Positive' if predictions[0] > 0.5 else 'Negative'

# Function to process the image and predict
def load_model_and_predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array /= 255.0  # Rescale
    return predict(img_array)

# GUI Functionality
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = load_model_and_predict(file_path)
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = Label(window, image=img)
        panel.image = img
        panel.pack()
        result_label.config(text=f'Result: {result}')

# Set up the GUI
window = Tk()
window.title('Liver Ultrasound Classification')

upload_button = Button(window, text='Upload Image', command=upload_image)
upload_button.pack()

result_label = Label(window, text='Result: ')
result_label.pack()

window.mainloop()
