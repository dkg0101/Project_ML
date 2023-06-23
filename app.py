
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)


@app.route("/" , methods=['GET','POST'])
def index():
    return 'starting machine learning project...'


if __name__ == "__main__":
    app.run(debug=True)
    
