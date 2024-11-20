from flask import Flask, render_template, url_for, Response, redirect, request, jsonify, session, flash, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename 
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from models.camera import Video
from models.dbconn import get_db_connection , close_db_connection


app = Flask(__name__)
CORS(app)


# Configure MySQL connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'SAS'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.secret_key = 'abc123'  # Used for session management

@app.teardown_appcontext
def teardown_db(exception):
    close_db_connection()


# Start New Code

# Load the trained recognizer model (train once and save the model)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to decode base64 image
def decode_base64_image(data):
    try:
        image_data = data.split(",")[1]
        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# Function to train the recognizer and save the model
def train_recognizer():
    db = get_db_connection()
    cursor = db.cursor()

    # Fetch all staff images from the database
    cursor.execute("SELECT id, img FROM staff_tbl")
    staff_records = cursor.fetchall()

    staff_images = []
    staff_ids = []

    for staff in staff_records:
        staff_id = staff['id']
        staff_image_blob = staff['img']

        # Convert BLOB data to numpy array and decode it
        staff_image_np = np.frombuffer(staff_image_blob, np.uint8)
        staff_image = cv2.imdecode(staff_image_np, cv2.IMREAD_COLOR)

        if staff_image is None:
            print(f"Error decoding staff image for ID: {staff_id}")
            continue

        # Convert staff image to grayscale and add to training data
        staff_gray_img = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)
        staff_images.append(staff_gray_img)
        staff_ids.append(staff_id)

    if staff_images and staff_ids:
        recognizer.train(staff_images, np.array(staff_ids))
        # Save the model
        recognizer.save('trained_recognizer.yml')
        print("Recognizer trained and model saved.")
    else:
        print("No staff data available for training.")

    cursor.close()


# Route to take attendance
@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    try:
        # train_recognizer()
        print("Entering take_attendance function")
        
        # Load the trained recognizer model
        recognizer.read('trained_recognizer.yml')

        # Get the image data from the request
        data = request.json.get('image')
        if not data:
            print("No image data received")
            return jsonify({'status': 'error', 'message': 'No image data received'})

        # Decode the base64 image to an OpenCV image
        img = decode_base64_image(data)
        if img is None:
            print("Failed to decode image")
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})

        # Convert the captured image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("No face detected")
            return jsonify({'status': 'error', 'message': 'No face detected'})

        db = get_db_connection()
        cursor = db.cursor()

        matched_staff = []  # Store matched staff IDs
        marked_attendance_ids = set()  # To avoid duplicate attendance marks

        # Loop through each detected face in the captured image
        for (x, y, w, h) in faces:
            face_roi = gray_img[y:y + h, x:x + w]

            # Predict if the captured face matches any staff image
            label, confidence = recognizer.predict(face_roi)
            print(f"Predicted label: {label}, Confidence: {confidence}")

            # Lower confidence threshold to allow for more flexibility
            threshold = 70  # Lowered from 70 to 55
            if confidence < threshold:
                print(f"Staff ID {label} found in records with confidence: {confidence}")

                # Mark attendance for the matched staff
                if label not in marked_attendance_ids:
                    matched_staff.append(label)
                    marked_attendance_ids.add(label)  # Add to the set of marked IDs

                    # Mark attendance for the matched staff
                    attendance_date = datetime.now().strftime('%Y-%m-%d')
                    
                    cursor.execute(""" 
                        INSERT INTO staffattendance (staff_id, attendance_date, status) 
                        VALUES (%s, %s, 'Present') 
                    """, (label, attendance_date))
                    db.commit()
                    # print(f"Attendance marked for staff ID: {label}")
            else:
                print(f"Staff ID {label} NOT found in records or confidence too low: {confidence}")

        cursor.close()

        if matched_staff:
            # If any matches were found, return success with the matched IDs
            return jsonify({'status': 'success', 'message': f'Attendance marked for staff IDs: {matched_staff}'})
        else:
            # If no matches found, return an error message
            return jsonify({'status': 'error', 'message': 'No matching staff found'})

    except Exception as e:
        print(f"Error in take_attendance: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred: ' + str(e)})


# End new code




# Home Page Route
@app.route('/')
def home():
    train_recognizer()
    return render_template('index.html')

# Home page camrea set
def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

# Home page camera accessing
@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
# login page Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        login_name = request.form['admin_id']
        password = request.form['password']
        
        # Connect to database
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Query to get admin details by login_name
        query = "SELECT * FROM admin WHERE login_name = %s"
        cursor.execute(query, (login_name,))
        admin = cursor.fetchone()

        if admin:
            # Extract the hashed password from the database
            stored_hashed_password = admin['Password']
            
            # Verify the entered password against the hashed password
            if check_password_hash(stored_hashed_password, password):
                # Password matches, login successful - store all admin details in session
                session['loggedin'] = True
                session['admin'] = {
                    'id': admin['id'],
                    'admin_name': admin['Name'],
                    'login_name': admin['Login_Name'],
                    'email': admin['Email'],
                    'Pno': admin['Pnumber']
                }

                # Redirect to the dashboard
                return redirect(url_for('dashboard'))
            else:
                # Password doesn't match
                flash('Invalid Login Name or Password', 'danger')
                return render_template('admin/login.html')
        else:
            # No user found with that login_name
            flash('Invalid Login Name or Password', 'danger')
            return render_template('admin/login.html')

    return render_template('admin/login.html')




# Dashboard Route
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    
    if 'loggedin' in session:
        
        # train_recognizer()
        
        # Establish the database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch record for count staff in frontEnd
        cursor.execute("SELECT * FROM staff_tbl")
        countstaff= cursor.fetchall()

        # Fetch attendance records for the current date with staff name and department
        cursor.execute("""
            SELECT * FROM staffattendance
            JOIN staff_tbl ON staffattendance.staff_id = staff_tbl.id
            WHERE staffattendance.attendance_date = CURDATE()
            GROUP BY staff_tbl.id, staffattendance.attendance_date
        """)
        staff_records = cursor.fetchall()
        for record in staff_records:
            record['attendance_date'] = record['attendance_date'].strftime('%d-%m-%Y')

        # Close cursor and connection
        absent_staff_count = len(countstaff) - len(staff_records)
        cursor.close()
        # conn.close()
        # Render the data in the template
        return render_template('admin/dashboard.html', staff =staff_records, countstaff = countstaff, absent = absent_staff_count)
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))



@app.route('/fullAttendanceRecord')
def fullAttendanceRecord():
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch record for count staff in frontEnd
        cursor.execute("SELECT * FROM staff_tbl")
        countstaff= cursor.fetchall()

        # Fetch attendance records for the current date with staff name and department
        cursor.execute("""
            SELECT * FROM staffattendance
            JOIN staff_tbl ON staffattendance.staff_id = staff_tbl.id
            GROUP BY staff_tbl.id, staffattendance.attendance_date
            ORDER BY staffattendance.attendance_date DESC;
        """)
        staff_records = cursor.fetchall()
        for record in staff_records:
            record['attendance_date'] = record['attendance_date'].strftime('%d-%m-%Y')

        # Close cursor and connection
        cursor.close()
        return render_template('admin/fullAttendanceRecord.html', staff =staff_records)
    else:
        # flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))

# GROUP BY staff_tbl.id, staffattendance.attendance_date
# Edit Attendance Record
@app.route('/AttendanceEdit/<int:id>', methods=['GET', 'POST'])
def AttendanceEdit(id):
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch attendance records for the current date with staff name and department
        cursor.execute("""
            SELECT * FROM staffattendance
            JOIN staff_tbl ON staffattendance.staff_id = staff_tbl.id
            ORDER BY staffattendance.attendance_date DESC;
        """)
        staff_records = cursor.fetchall()
        
        for record in staff_records:
            record['attendance_date'] = record['attendance_date'].strftime('%d-%m-%Y')
        
        # Extract the name from the first staff record, if available
        staff_name = staff_records[0]['name'] if staff_records else None

        # Close cursor and connection
        cursor.close()
        return render_template('admin/AttendanceEdit.htm', staff_name=staff_name, staff=staff_records)
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))



# Delete One Attendance Record
@app.route('/deleteattendance/<int:id>', methods=['POST'])
def deleteattendance(id):
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM staffattendance WHERE attendance_id = %s', (id,))
        # cursor.execute('DELETE FROM staff_tbl WHERE id = %s', (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash('Attendance record deleted successfully!', 'success')
        return redirect(url_for('fullAttendanceRecord'))
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))
  

# Delete All Attendance Record 
@app.route('/deleteAllattendance/<int:id>', methods=['POST'])
def deleteAllattendance(id):
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM staffattendance WHERE staff_id = %s', (id,))
        # cursor.execute('DELETE FROM staff_tbl WHERE id = %s', (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash('Attendance record deleted successfully!', 'success')
        return redirect(url_for('fullAttendanceRecord'))
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))



# Edit Profile Route
@app.route('/editAdminprofile', methods=['GET', 'POST'])
def editAdminprofile():
    if 'loggedin' in session:
        admin_id = session['admin']['id']

        if request.method == 'POST':
            admin_name = request.form.get('admin_name')
            login_name = request.form.get('login_name')
            email = request.form.get('email')
            Pno = request.form.get('Pno')

            # Validate the received form data
            if not all([admin_name, login_name, email, Pno]):
                flash('All fields are required', 'warning')
                return redirect(url_for('editAdminprofile'))

            # Handle the image upload
            img_file = request.files.get('img')  # Get the uploaded file
            
            # Validate image file
            img_data = None
            if img_file and img_file.filename != '':
                img_data = img_file.read()  # Read the image file data
                print("Image uploaded successfully.")  # Debugging
            else:
                print("No image uploaded, using existing image.")  # Debugging

            print("Received form data:", admin_name, login_name, email, Pno)  # Debugging

            # Update the admin details in the database
            try:
                db = get_db_connection()
                cursor = db.cursor()

                # Update query to modify admin details based on the admin_id
                if img_data:  # If new image data is provided
                    cursor.execute(""" 
                        UPDATE admin 
                        SET Name = %s, Login_Name = %s, Email = %s, Pnumber = %s, img = %s
                        WHERE id = %s
                    """, (admin_name, login_name, email, Pno, img_data, admin_id))
                else:  # If no new image is provided, exclude the img field
                    cursor.execute(""" 
                        UPDATE admin 
                        SET Name = %s, Login_Name = %s, Email = %s, Pnumber = %s
                        WHERE id = %s
                    """, (admin_name, login_name, email, Pno, admin_id))

                db.commit()  # Commit the changes to the database
                cursor.close()

                # Update session data with the new values
                session['admin']['admin_name'] = admin_name
                session['admin']['login_name'] = login_name
                session['admin']['email'] = email
                session['admin']['Pno'] = Pno
                if img_data:
                    session['admin']['img_data'] = base64.b64encode(img_data).decode('utf-8')

                flash('Profile updated successfully!', 'success')
                return redirect(url_for('editAdminprofile'))  # Reload the page after update
            except Exception as e:
                print(f"Error updating profile: {e}")  # Debugging
                flash(f"Error updating profile: {str(e)}", 'danger')
                return redirect(url_for('editAdminprofile'))  # Stay on the same page on error
        else:
            # Render the form with current session data
            return render_template('admin/editAdminprofile.html')
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))




# Change Password
@app.route('/changePassword', methods=['GET', 'POST'])
def changePassword():
    if 'loggedin' in session:
        admin_id = session['admin']['id']  # Admin's ID from session
        print(f"Admin ID: {admin_id}")  # Debugging: Check the admin ID

        if request.method == 'POST':
            current_password = request.form.get('password')  # Current password input
            new_password = request.form.get('newpassword')   # New password input
            renew_password = request.form.get('renewpassword')  # Re-enter new password input

            # Check if new passwords match
            if new_password != renew_password:
                flash('New password and confirmation password do not match', 'warning')
                return redirect(url_for('changePassword'))

            try:
                db = get_db_connection()
                cursor = db.cursor()

                # Fetch the current password from the database for the logged-in admin
                cursor.execute("SELECT Password FROM admin WHERE id = %s", (admin_id,))
                result = cursor.fetchone()

                if result:
                    stored_password = result['Password']  # Access the password from the tuple
                    print(f"Stored password from DB: {stored_password}")  # Debugging: Check fetched password
                    print(f"Entered current password: {current_password.strip()}")  # Debugging: Check input password

                    # Compare current password input with the stored password (if it is plain text for now)
                    if check_password_hash(stored_password, current_password.strip()):
                        print("Passwords match!")  # Debugging: Confirm password match

                        # Hash the new password before storing
                        new_password_hash = generate_password_hash(new_password)
                        print(f"Hashed new password: {new_password_hash}")  # Debugging: Check hashed password

                        # Update the password in the database with the new hashed password
                        cursor.execute("UPDATE admin SET Password = %s WHERE id = %s", (new_password_hash, admin_id))

                        # Print the SQL query for debugging
                        print(f"SQL Query: UPDATE admin SET Password = '{new_password_hash}' WHERE id = {admin_id}")

                        # Print row count affected for debugging
                        print(f"Rows affected: {cursor.rowcount}")  # Debugging: Check row count

                        # Commit the changes to the database
                        db.commit()

                        # Check if the commit was successful
                        if cursor.rowcount > 0:
                            flash('Password updated successfully!', 'success')
                            return redirect(url_for('editAdminprofile'))
                        else:
                            flash('No changes were made. Please try again.', 'danger')
                            return redirect(url_for('changePassword'))
                    else:
                        flash('Current password is incorrect', 'danger')
                        return redirect(url_for('changePassword'))
                else:
                    flash('User not found', 'danger')
                    return redirect(url_for('changePassword'))

            except Exception as e:
                # Print the exact exception message to understand the problem
                print(f"Error changing password: {e}")  # Debugging: print the actual error
                flash(f"Error changing password: {str(e)}", 'danger')
                return redirect(url_for('changePassword'))

        return render_template('admin/editAdminprofile.html')  # Ensure you have a separate template for password changes
    else:
        flash('Please log in to change the password', 'warning')
        return redirect(url_for('login'))




# Login Route
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('admin_name', None)
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))




# Staff page Route
@app.route('/staffrecord')
def staffrecord():
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM staff_tbl")
        staff_records = cursor.fetchall()
        cursor.close()
        # conn.close()
        return render_template('admin/StaffRecord.html', staff=staff_records)
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))




# edit Staff page Route
@app.route('/edituser/<int:id>', methods=['GET', 'POST'])
def edituser(id):
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Fetch the staff record from the database
        cursor.execute("SELECT * FROM staff_tbl WHERE id = %s", (id,))
        staff_record = cursor.fetchone()
        
        if staff_record and staff_record['img']:
            # Convert binary image data to Base64
            staff_record['img'] = base64.b64encode(staff_record['img']).decode('utf-8')
        
        if request.method == 'POST':
            # Get the updated data from the form
            name = request.form['name']
            designation = request.form['designation']
            department = request.form['department']
            phoneNo = request.form['phoneNo']
            email = request.form['email']
            address = request.form['address']
            
            # Check if a new image has been uploaded
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file.filename != '':
                    img_data = image_file.read()  # Read binary data from the uploaded image
                    # Update the staff details with a new image
                    cursor.execute("""
                        UPDATE staff_tbl
                        SET name = %s, designation = %s, department = %s, phoneNo = %s, email = %s, address = %s, img = %s
                        WHERE id = %s
                    """, (name, designation, department, phoneNo, email, address, img_data, id))
                else:
                    # Update without image if no new image is uploaded
                    cursor.execute("""
                        UPDATE staff_tbl
                        SET name = %s, designation = %s, department = %s, phoneNo = %s, email = %s, address = %s
                        WHERE id = %s
                    """, (name, designation, department, phoneNo, email, address, id))
            else:
                # Update without image if no image field is present
                cursor.execute("""
                    UPDATE staff_tbl
                    SET name = %s, designation = %s, department = %s, phoneNo = %s, email = %s, address = %s
                    WHERE id = %s
                """, (name, designation, department, phoneNo, email, address, id))

            
            conn.commit()  # Save changes
            cursor.close()
            conn.close()
            flash('Staff record updated successfully!', 'success')
            return redirect(url_for('staffrecord'))  # Redirect to the staff list page
                
        return render_template('admin/edituser.html', staff = staff_record)  # Path to the dashboard page within 'admin' folder
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))



# Add Staff page Route
@app.route('/addstaff', methods=['GET', 'POST'])
def addstaff():
    if 'loggedin' in session:
        if request.method == 'POST':
            name = request.form['name']
            department = request.form['department']
            designation = request.form['designation']
            phoneNo = request.form['phoneNo']
            email = request.form['email']
            img = request.files['img'].read()
            address = request.form['address']
            time = datetime.now()        

            # Insert data into the database
            try:
                db = get_db_connection()
                cursor = db.cursor()
                sql = '''
                    INSERT INTO staff_tbl (name, designation, department, phoneNo, email, img, address, time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                '''
                cursor.execute(sql, (name, designation, department, phoneNo, email, img, address, time))

                # Commit the transaction
                db.commit()

                flash('Staff member added successfully!', 'success')
                return redirect(url_for('addstaff'))  # Redirect after successful addition
            except Exception as e:
                db.rollback()  # Rollback in case of an error
                flash(f'Error: {str(e)}', 'danger')
            finally:
                cursor.close()
                close_db_connection()
        
        # Render the add staff form
        return render_template('admin/addstaff.html')
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))


@app.route('/deletestaff/<int:id>', methods=['POST'])
def deletestaff(id):
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM staffattendance WHERE staff_id = %s', (id,))
        cursor.execute('DELETE FROM staff_tbl WHERE id = %s', (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash('Staff member deleted successfully!', 'success')
        return redirect(url_for('staffrecord'))
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))




# Show Staff Members Route
@app.route('/stafflist', methods=['GET', 'POST'])
def stafflist():
    if 'loggedin' in session:
        # Fetch all staff records from the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM staff_tbl")
        staff_records = cursor.fetchall()
        cursor.close()
        # conn.close()
        
        # Convert binary image data to Base64 for each staff member
        for staff in staff_records:
            if staff['img']:  # Check if there is image data
                # Convert binary data to Base64
                staff['img'] = base64.b64encode(staff['img']).decode('utf-8')
                
        return render_template('admin/stafflist.html', staff=staff_records)
    else:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))


# Working page Route
# @app.route('/workingPage')
# def workingPage():
#     if 'loggedin' in session:
#         return render_template('admin/pages-error-404.html')
#     else:
#         flash('Please log in to access the dashboard', 'warning')
#         return redirect(url_for('login'))
 
#  About US page
# @app.route('/AboutUS')
# def AboutUS():
#         return render_template('AboutUs.html')


# close the DB connection 
@app.teardown_appcontext
def teardown_db(exception):
    close_db_connection()


if __name__ == '__main__':
    with app.app_context():
        train_recognizer()
    app.run(debug=True)