from flask import Flask
from app.api.v1.openai_endpoints import api_blueprint

flaskApp = Flask(__name__)

# Registrar o Blueprint
flaskApp.register_blueprint(api_blueprint)

if __name__ == "__main__":
    flaskApp.run(host="0.0.0.0", port=8000, debug=True)
