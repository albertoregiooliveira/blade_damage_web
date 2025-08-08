from flask import render_template

from src.core.upload import upload_file

def setup_routes(app):

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        return upload_file()
