from flask import Flask, render_template, request, jsonify, session
from bot_backend import get_bot_response

app = Flask(__name__)
app.secret_key = "cyberlaw_secret"

@app.route("/")
def index():
    # Clear chat on refresh
    session.pop("history", None)
    session["history"] = []
    return render_template("index.html", chat_history=session["history"])

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json["message"]
    bot_msg = get_bot_response(user_msg)

    entry = {"user": user_msg, "bot": bot_msg}
    history = session.get("history", [])
    history.append(entry)
    session["history"] = history

    return jsonify({"bot": bot_msg})

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)
