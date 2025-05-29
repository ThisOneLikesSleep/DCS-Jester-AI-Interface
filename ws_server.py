import asyncio
import websockets

async def handle_connection(websocket, path, sock_j_outbound, sock_j_inbound):
    print("WS client connected")

    async def receive_messages():
        async for message in websocket:
            print(f"Received message: {message}")
            sock_j_inbound.put(message)

    async def send_messages():
        while True:
            # Check if there's an item in the queue
            if not sock_j_outbound.empty():
                item = sock_j_outbound.get()
                await websocket.send(item)
                print(f"Sent message: {item}")
            await asyncio.sleep(0.1)

    # Run both receiving and sending concurrently
    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_messages())

    # Wait for both tasks to complete (they won't in this infinite loop scenario)
    await asyncio.gather(receive_task, send_task)

def start_ws(sock_j_outbound, sock_j_inbound):
    start_server = websockets.serve(
        lambda ws, path: handle_connection(ws, path, sock_j_outbound, sock_j_inbound),
        "localhost", 6216
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    print("WebSocket server started on ws://localhost:6216")
    asyncio.get_event_loop().run_forever()