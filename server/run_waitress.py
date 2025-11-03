from waitress import serve
from server.init_routes import create_app
import argparse

app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"[INFO] Starting Waitress on port {args.port}...")
    serve(app, host='localhost', port=args.port, threads=20)