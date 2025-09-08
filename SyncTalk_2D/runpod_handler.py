import runpod
import os
import time
import threading
import uvicorn
from synctalk_fastapi import app, logger
import requests

def check_server_status():
    """Check if the job is ready to be marked as completed and reset the flag if so"""
    try:
        response = requests.get("http://localhost:8000/termination_status?reset=true")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Checking job status: {data['is_job_terminated']}")
            if data['is_job_terminated']:
                logger.info("Termination flag detected - exiting")
                return True
        return False
    except requests.RequestException as e:
        logger.error(f"Error checking termination status: {e}")
        return False

def keep_waiting():
    """Wait until job is ready to be marked as completed"""
    # Check status every 10 seconds
    while True:
        if check_server_status():
            return True
        time.sleep(10)

def handler(event):
    """
    Runpod handler function
    """
    global is_job_terminated
    
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP", "localhost")
    tcp_port = int(os.environ.get("RUNPOD_TCP_PORT_8000", 8000))

    print(f"Public IP: {public_ip}")
    print(f"TCP Port: {tcp_port}")

    runpod.serverless.progress_update(event, {
        "message": f"Server running at {public_ip}:{tcp_port}",
        "public_ip": public_ip,
        "tcp_port": tcp_port
    })
    
    # Wait for the server to be ready with the specified conditions
    result = keep_waiting()
    
    runpod.serverless.progress_update(event, {
        "message": "Job is ready to be marked as completed",
        "public_ip": public_ip,
        "tcp_port": tcp_port
    })

    return {
        "message": "completed",
        "public_ip": public_ip,
        "tcp_port": tcp_port
    }

if __name__ == "__main__":
    # Start the uvicorn server in a separate thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000)
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Start the runpod handler
    runpod.serverless.start({"handler": handler})