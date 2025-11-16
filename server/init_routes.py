from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# Allow all origins for /node/* (good for dev; tighten later for prod)
CORS(app, resources={r"/node/*": {"origins": "*"}})

def create_app():
    # Import and register blueprints
    from route.nodes_route import nodes_bp
    from route.map_route import floodwatch_bp
    app.register_blueprint(nodes_bp)
    app.register_blueprint(floodwatch_bp)
    return app