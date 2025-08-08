from flask import Flask
from src.core.routes import setup_routes


# Método principal
if __name__ == "__main__":
    app = Flask(__name__)
    setup_routes(app)
    app.run(debug=True)