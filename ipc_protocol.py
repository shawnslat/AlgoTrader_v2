"""
IPC Protocol for Trading Bot GUI Communication
Uses Unix domain sockets for local inter-process communication
"""
import json
import socket
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/trader_bot.sock"


class IPCServer:
    """Server that handles incoming commands from the GUI"""

    def __init__(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.socket_path = SOCKET_PATH
        self.handler = handler
        self.server_socket = None
        self.running = False
        self.thread = None

    def start(self):
        """Start the IPC server in a background thread"""
        # Remove existing socket file if present
        if Path(self.socket_path).exists():
            Path(self.socket_path).unlink()

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.running = True

        self.thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.thread.start()
        logger.info(f"IPC server started on {self.socket_path}")

    def _accept_connections(self):
        """Accept and handle client connections"""
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")

    def _handle_client(self, client_socket):
        """Handle a single client connection"""
        try:
            # Receive data
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Check if we have a complete message (ends with newline)
                if b"\n" in chunk:
                    break

            if not data:
                return

            # Parse command
            message = json.loads(data.decode('utf-8').strip())
            logger.debug(f"Received command: {message.get('command')}")

            # Handle command
            response = self.handler(message)

            # Send response
            response_data = json.dumps(response).encode('utf-8') + b"\n"
            try:
                client_socket.sendall(response_data)
            except (BrokenPipeError, ConnectionResetError) as e:
                # Client disconnected before response sent - not an error
                logger.debug(f"Client disconnected during response: {e}")
            except Exception as e:
                logger.error(f"Error sending response: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
            error_response = json.dumps({"error": str(e)}).encode('utf-8') + b"\n"
            try:
                client_socket.sendall(error_response)
            except (BrokenPipeError, ConnectionResetError):
                # Client already disconnected
                pass
            except Exception:
                pass
        finally:
            client_socket.close()

    def stop(self):
        """Stop the IPC server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if Path(self.socket_path).exists():
            Path(self.socket_path).unlink()
        logger.info("IPC server stopped")


class IPCClient:
    """Client for sending commands to the bot service"""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path

    def send_command(self, command: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        """Send a command and wait for response"""
        try:
            # Connect to server
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            client_socket.connect(self.socket_path)

            # Send command
            message = json.dumps(command).encode('utf-8') + b"\n"
            client_socket.sendall(message)

            # Receive response
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break

            client_socket.close()

            if not data:
                return {"error": "No response from server"}

            response = json.loads(data.decode('utf-8').strip())
            return response

        except socket.timeout:
            return {"error": "Request timed out"}
        except FileNotFoundError:
            return {"error": "Bot service not running"}
        except ConnectionRefusedError:
            # Socket file may exist but no process is listening (stale socket).
            try:
                if Path(self.socket_path).exists():
                    Path(self.socket_path).unlink()
            except Exception:
                pass
            return {"error": "Bot service not running (connection refused)"}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"error": str(e)}

    def is_running(self) -> bool:
        """Check if the bot service is running"""
        if not Path(self.socket_path).exists():
            return False
        # Detect stale sockets (file exists but nothing is listening).
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(self.socket_path)
            s.close()
            return True
        except Exception:
            try:
                Path(self.socket_path).unlink()
            except Exception:
                pass
            return False


class LogStreamer:
    """Stream log entries to connected clients"""

    def __init__(self):
        self.subscribers = []
        self.lock = threading.Lock()

    def subscribe(self, callback: Callable[[str], None]):
        """Subscribe to log stream"""
        with self.lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]):
        """Unsubscribe from log stream"""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def emit(self, log_entry: str):
        """Emit a log entry to all subscribers"""
        with self.lock:
            for callback in self.subscribers[:]:  # Copy list to avoid modification during iteration
                try:
                    callback(log_entry)
                except Exception as e:
                    logger.error(f"Error in log subscriber: {e}")
