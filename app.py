import asyncio
import uvicorn
import cv2
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Dict, Any

try:
    from drone_manager import drone_manager
except ImportError as e:
    print(f"[ERROR] drone_manager.py 또는 detector.py를 찾을 수 없습니다. {e}")
    exit()


# 서버 관리
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[Server] Starting up...")
    asyncio.create_task(connect_drone_async())

    broadcast_task = asyncio.create_task(broadcast_telemetry_loop())
    yield

    print("[Server] Shutting down...")

    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    drone_manager.shutdown()


async def connect_drone_async():

    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, drone_manager.connect)
    if not success:
        print("[Server] Failed to connect to Drone/Webcam.")


app = FastAPI(lifespan=lifespan)


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    print("[WARN] 'static' directory not found.")


# 웹소켓 연결
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        message_str = json.dumps(message)

        connections_to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                connections_to_remove.append(connection)

        for connection in connections_to_remove:
            self.disconnect(connection)


manager = ConnectionManager()


# 텔레메트리 전송
async def broadcast_telemetry_loop():
    print("[Server] Telemetry broadcast loop started.")
    while True:
        try:
            telemetry_data = drone_manager.get_latest_telemetry()
            await manager.broadcast(telemetry_data)
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("[Server] Telemetry broadcast loop cancelled.")
            break
        except Exception as e:
            print(f"[Server] Error in telemetry loop: {e}")
            await asyncio.sleep(1.0)


@app.get("/")
async def get():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>Error: static/index.html not found</h1>", status_code=404
        )


async def generate_mjpeg_stream():

    while True:
        frame = drone_manager.get_latest_frame()
        if frame is not None:
            ret, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        await asyncio.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_message(message)
            except json.JSONDecodeError:
                print(f"[Server] Received invalid JSON: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("[Server] Client disconnected from /ws/data")
    except Exception as e:
        print(f"[Server] Error in websocket handler: {e}")
        manager.disconnect(websocket)


async def handle_message(message: Dict[str, Any]):
    msg_type = message.get("type")

    if msg_type == "control_command":
        command = message.get("command")
        data = message.get("data")
        if command:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, drone_manager.send_command, command, data)


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
