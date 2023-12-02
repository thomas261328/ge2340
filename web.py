from detector.detector import Detector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import threading
import cv2
import os

dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(dirname, 'images')
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
exit_event= threading.Event()

outputFrame = None
lock = threading.Lock()

app = Flask(__name__, static_url_path="/templates", static_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()

@app.route("/")
def index():
	
	return render_template("index.html")

def face_rec():

	global vs, outputFrame, lock
	process_this_frame = True

	md = Detector()
	md.inital_load()

	while True:

		frame = vs.read()
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

		if process_this_frame:
			face_locations, face_names = md.face_rec(rgb_small_frame)
		
		process_this_frame = not process_this_frame

		# Display the results
		for (top, right, bottom, left), name in zip(face_locations, face_names):
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

		with lock:
			outputFrame = frame.copy()


def generate():

	global outputFrame, lock
	while True:
		with lock:

			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")



@app.route('/upload', methods=['POST'])
def upload_file():
	if 'file' in request.files:
		name = request.args.get('name')
		file = request.files['file']
		print(name)
		print(file.filename)
		
		if file and Detector.allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return 'File uploaded successfully'
	return 'No file uploaded'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':

	hostname = "localhost"
	port = "8080"

	t = threading.Thread(target=face_rec)
	t.start()

	# set the project root directory as the static folder, you can set others.
	app.run(host=hostname, port=port, debug=True,
		threaded=True, use_reloader=False)

vs.stop()