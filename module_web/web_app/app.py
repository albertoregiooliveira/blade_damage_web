from flask import Flask
from module_web.web_app.src.core.routes import setup_routes

app = Flask(__name__)
setup_routes(app)

# Método principal
if __name__ == "__main__":
    app.run(debug=True)