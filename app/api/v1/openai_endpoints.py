from flask import Blueprint, jsonify, request
from app.core.openai_client import handle_chatbot_response
from app.utils.helpers import validate_message

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/", methods=["GET"])
def read_root():
    return jsonify({"status": "Chatbot API is up and running"})


@api_blueprint.route("/api/v1/send_message/", methods=["POST"])
def send_message():
    try:
        data = request.json
        user_message = validate_message(data)
        response = handle_chatbot_response(user_message)
        return jsonify({"response": response})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
