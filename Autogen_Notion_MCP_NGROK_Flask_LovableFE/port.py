from flask import Flask,jsonify
import os

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': "Hello, World!"})


if __name__ == '__main__':
    port = 7001
    os.environ['FLASK_ENV'] = 'development'


    app.run(port=port)
