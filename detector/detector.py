import numpy as np
import imutils
import cv2
import os
import face_recognition

IMAGES_FOLDER = "images/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

class Detector():

	def __init__(self):
		self.known_face_encodings = []
		self.known_face_names = []
		
	def allowed_file(filename):
		return ('.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS)

	def inital_load(self):
		
		images_file = []
		for filename in os.listdir(IMAGES_FOLDER):
			
			if Detector.allowed_file(filename): 
				images_file.append(filename)

		for image_file in images_file:
			# Load a second sample picture and learn how to recognize it.
			image_filename = os.path.splitext(image_file)[0]
			image_file = IMAGES_FOLDER + image_file
			image = face_recognition.load_image_file(image_file)
			image_face_encoding = face_recognition.face_encodings(image)[0]
			self.known_face_encodings.append(image_face_encoding)
			self.known_face_names.append(image_filename)

	def face_rec(self, rgb_small_frame):

		face_locations = []
		face_encodings = []
		face_names = []

		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
			name = "Unknown"

			face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = self.known_face_names[best_match_index]

			face_names.append(name)
		
		return face_locations, face_names
