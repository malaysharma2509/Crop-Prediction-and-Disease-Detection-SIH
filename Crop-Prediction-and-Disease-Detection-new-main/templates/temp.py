@app.route('/crop_disease', methods=['GET', 'POST'])
def crop_disease():
    if request.method == 'POST':
        file = request.files.get('file')  # Using .get() for safety

        if file:
            # Secure the filename
            filename = secure_filename(file.filename)

            # Create the uploads directory if it doesn't exist
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)  # This will create the folder if it does not exist

            # Save the file
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Open the image and preprocess it
            image = Image.open(file_path)  # Open the saved file
            img_batch = preprocess_image(image)  # Ensure this function is defined properly

            # Make prediction
            predictions = disease_model.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions) * 100  # Convert to percentage

            # Return results to the template
            return render_template("crop_disease2.html", prediction=predicted_class, confidence=confidence, filename=filename)

    return render_template("crop_disease2.html", prediction=None)