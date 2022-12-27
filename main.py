from flask import Flask, request
import numpy as np
import os
import pyboof as pb

app = Flask(__name__)


@app.route('/api', methods=['GET'])
def api():
    return 'Hello World!'


@app.route('/api/microqr-decoder', methods=['POST'])
def microqr_decoder():
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Generate a random file name with the same extension
    filename = os.urandom(24).hex() + os.path.splitext(file.filename)[1]

    file.save(os.path.join('uploads', filename))

    detector = pb.FactoryFiducial(np.uint8).microqr()

    image = pb.load_single_band('uploads/' + filename, np.uint8)

    detector.detect(image)

    if len(detector.detections) == 0:
        return {"message": "No Micro QR detected"}

    # Delete the file after processing
    os.remove('uploads/' + filename)

    return {
        "message": "QR code detected",
        "data": detector.detections[0].message
    }


print('Starting server...')
if __name__ == '__main__':
    app.run(debug=True)

# how to run the app
# py -m flask run
