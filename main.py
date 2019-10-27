from flask import Flask, render_template, Response
from camera import VideoCamera

import queue
import threading

Q = queue.Queue()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        yield Q.get()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    def run_app():
        app.run(host='0.0.0.0', debug=False, threaded=False)
    x = threading.Thread(target=run_app)
    x.start()
    v = VideoCamera()
    while True:
        frame = v.get_frame()
        Q.put(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
