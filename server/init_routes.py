from flask import Flask

app = Flask(__name__)

def create_app():
    # Import and register blueprints
    from route.nodes_route import nodes_bp
    app.register_blueprint(nodes_bp)
    return app