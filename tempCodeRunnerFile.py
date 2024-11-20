@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    try:
        # Get the image data from the request
        data = request.json.get('image')
        if not data:
            return jsonify({'status': 'error', 'message': 'No image data received'})

        # Decode the base64 image to an OpenCV image
        img = decode_base64_image(data)
        if img is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})

        # Convert to grayscale (if needed for face recognition)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        # Check if at least one face was detected
        if len(faces) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected'})

        # Connect to the database
        db = get_db_connection()
        cursor = db.cursor()

        # Get the staff images from the database
        cursor.execute("SELECT * FROM staff_tbl")
        staff_records = cursor.fetchall()

        # Match the detected face with database images
        for staff in staff_records:
            staff_id = staff['id']  # Assuming ID is in the first column
            staff_image_blob = staff['img']  # Assuming BLOB image is in the second column

            # Convert BLOB data to numpy array and decode it
            staff_image_np = np.frombuffer(staff_image_blob, np.uint8)
            staff_image = cv2.imdecode(staff_image_np, cv2.IMREAD_COLOR)

            # Ensure the staff image was loaded properly
            if staff_image is None:
                print(f"Error decoding staff image for ID: {staff_id}")
                continue  # Skip to the next staff if there’s an error

            # Convert to grayscale
            staff_gray_img = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)

            # Use template matching
            result = cv2.matchTemplate(gray_img, staff_gray_img, cv2.TM_CCOEFF_NORMED)

            # If there's a match (with a confidence threshold)
            threshold = 0.7
            if np.max(result) >= threshold:
                # Mark attendance
                attendance_date = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    INSERT INTO staffattendance (staff_id, attendance_date)
                    VALUES (%s, %s)
                """, (staff_id, attendance_date))
                db.commit()  # Commit changes to the database
                cursor.close()
                db.close()  # Close the database connection
                return jsonify({'status': 'success', 'message': 'Attendance marked for staff ID: ' + str(staff_id)})

        cursor.close()
        db.close()  # Close the database connection
        return jsonify({'status': 'error', 'message': 'No matching staff found'})

    except Exception as e:
        print(f"Error in take_attendance: {e}")  # Print the error to the console for debugging
        return jsonify({'status': 'error', 'message': 'An error occurred: ' + str(e)})
