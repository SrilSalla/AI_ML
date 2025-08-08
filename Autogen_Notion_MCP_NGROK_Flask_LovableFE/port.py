from flask import Flask,jsonify
import os
from pyngrok import ngrok
from flask_cors import CORS

NGROK_AUTH_TOKEN = '30tV6mN2u7lhOTM432D5GToBw31_5QbAi3RB5XZb7Rw8bJTfi'

app = Flask(__name__)
CORS(app)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': "Hello, World!"})


if __name__ == '__main__':
    port = 7001
    os.environ['FLASK_ENV'] = 'development'

    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(port)
    print(f' * ngrok tunnel "{public_url}" -> "http://localhost:{port}"')

    app.run(port=port)
