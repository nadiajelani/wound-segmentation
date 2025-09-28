#!/usr/bin/env python3
"""
Start a simple HTTP server to serve the Wound Whisperer GUI
"""
import http.server
import socketserver
import webbrowser
import os
import sys

def start_gui_server(port=3000):
    """Start the GUI server"""
    
    # Change to the web directory
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    os.chdir(web_dir)
    
    # Create HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Wound Whisperer GUI starting...")
            print(f"📱 Open your browser to: http://localhost:{port}")
            print(f"📁 Serving from: {web_dir}")
            print(f"🔗 API Endpoint: http://127.0.0.1:8000")
            print(f"⏹️  Press Ctrl+C to stop")
            print("-" * 50)
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{port}')
                print("🚀 Browser opened automatically")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"   Please manually open: http://localhost:{port}")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 GUI server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python start_gui.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    start_gui_server(port)