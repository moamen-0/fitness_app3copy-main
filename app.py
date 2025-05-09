from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import os
import json
import pygame
import time

import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading
from flask_cors import CORS
import base64
import traceback
import eventlet
# Import exercise modules
from utils import calculate_angle, mp_pose, pose
from exercises.bicep_curl import hummer
from exercises.front_raise import dumbbell_front_raise
from exercises.squat import squat
from exercises.triceps_extension import triceps_extension
from exercises.lunges import lunges
from exercises.shoulder_press import shoulder_press
from exercises.plank import plank
from exercises.lateral_raise import side_lateral_raise
from exercises.triceps_kickback import triceps_kickback_side
from exercises.push_ups import push_ups

eventlet.monkey_patch()

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # For testing
    async_mode='eventlet',  # Important for WebSockets
    ping_timeout=60,
    ping_interval=25,
    logger=True,
    engineio_logger=True
)

# Setup for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Dummy sound class to avoid file path issues
class DummySound:
    def __init__(self):
        self.is_playing = False
    
    def play(self):
        self.is_playing = True
        print("Dummy sound play")
    
    def stop(self):
        self.is_playing = False
        print("Dummy sound stop")

# Use dummy sound instead of loading from a specific path
sound = DummySound()

# Ensure audio directory exists
os.makedirs("audio", exist_ok=True)

# Dictionary to store active sessions
active_sessions = {}

# Dictionary to store exercise functions
exercise_map = {
    'hummer': hummer,
    'front_raise': dumbbell_front_raise,
    'squat': squat,
    'triceps': triceps_extension,
    'lunges': lunges,
    'shoulder_press': shoulder_press,
    'plank': plank,
    'side_lateral_raise': side_lateral_raise,
    'triceps_kickback_side': triceps_kickback_side,
    'push_ups': push_ups
}

def get_valid_exercises():
    """Return a list of valid exercise IDs"""
    return list(exercise_map.keys())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/routes')
def api_routes():
    routes_info = []
    for rule in app.url_map.iter_rules():
        routes_info.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "url": str(rule)
        })
    return jsonify({
        "status": "AI Fitness Trainer API is running",
        "registered_routes": len(routes_info),
        "routes": routes_info
    })

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/exercise/<exercise>')
def exercise_page(exercise):
    """
    Unified endpoint for all exercise viewing modes
    """
    valid_exercises = get_valid_exercises()
    
    if exercise not in valid_exercises:
        app.logger.error(f"Invalid exercise requested: {exercise}")
        return "Exercise not found", 404
        
    return render_template('websocket_exercise.html', exercise_id=exercise)

@app.route('/api/exercises', methods=['GET'])
def get_exercises():
    """
    Return a list of available exercises
    """
    try:
        exercises = [
            {"id": "hummer", "name": "Bicep Curl (Hammer)"},
            {"id": "front_raise", "name": "Dumbbell Front Raise"},
            {"id": "squat", "name": "Squat"},
            {"id": "triceps", "name": "Triceps Extension"},
            {"id": "lunges", "name": "Lunges"},
            {"id": "shoulder_press", "name": "Shoulder Press"},
            {"id": "plank", "name": "Plank"},
            {"id": "side_lateral_raise", "name": "Side Lateral Raise"},
            {"id": "triceps_kickback_side", "name": "Triceps Kickback (Side View)"},
            {"id": "push_ups", "name": "Push Ups"}
        ]
        return jsonify(exercises)
    except Exception as e:
        app.logger.error(f"Error in get_exercises: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def api_status():
    """
    Simple endpoint to check API health
    """
    return jsonify({
        "status": "online",
        "timestamp": time.time(),
        "exercise_count": len(exercise_map)
    })

@app.route('/camera_test')
def camera_test():
    """
    Simple page to test camera access
    """
    return render_template('camera_test.html')

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    """
    MJPEG stream endpoint for direct video access
    """
    try:
        if exercise in exercise_map:
            # Add cache control headers
            return Response(
                exercise_map[exercise](sound), 
                mimetype='multipart/x-mixed-replace; boundary=frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        else:
            return "Invalid exercise", 400
    except Exception as e:
        app.logger.error(f"Error in video_feed: {str(e)}")
        app.logger.error(traceback.format_exc())
        return "Error processing video", 500

# ====================== WebSocket Event Handlers ======================

@socketio.on('connect')
def handle_connect():
    """
    Handle WebSocket connection
    """
    print(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'session_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle WebSocket disconnection and cleanup resources
    """
    print(f"Client disconnected: {request.sid}")
    # Clean up any active session on disconnect
    if request.sid in active_sessions:
        session_data = active_sessions[request.sid]
        if 'stop_event' in session_data:
            session_data['stop_event'].set()
        if 'cap' in session_data and session_data['cap'] is not None:
            session_data['cap'].release()
        del active_sessions[request.sid]

@socketio.on('start_exercise')
def handle_start_exercise(data):
    """
    Start exercise tracking via WebSocket
    """
    try:
        exercise_id = data.get('exercise_id')
        print(f"Starting exercise: {exercise_id} for session {request.sid}")
        
        if not exercise_id or exercise_id not in exercise_map:
            emit('error', {'message': f'Invalid exercise: {exercise_id}'})
            return
        
        # Stop any currently active session
        if request.sid in active_sessions:
            session_data = active_sessions[request.sid]
            if 'stop_event' in session_data:
                session_data['stop_event'].set()
            if 'cap' in session_data and session_data['cap'] is not None:
                session_data['cap'].release()
        
        # Create a stop event to allow safe termination
        stop_event = threading.Event()
        
        # Store session data
        active_sessions[request.sid] = {
            'exercise_id': exercise_id,
            'stop_event': stop_event,
            'cap': None,
            'left_counter': 0,
            'right_counter': 0,
            'start_time': time.time()
        }
        
        # Start a thread to process the exercise
        exercise_thread = threading.Thread(
            target=process_exercise_frames,
            args=(request.sid, exercise_id, stop_event)
        )
        exercise_thread.daemon = True
        exercise_thread.start()
        
        emit('exercise_started', {'exercise_id': exercise_id})
        
    except Exception as e:
        print(f"Error starting exercise: {str(e)}")
        traceback.print_exc()
        emit('error', {'message': f'Error starting exercise: {str(e)}'})

@socketio.on('stop_exercise')
def handle_stop_exercise():
    """
    Stop exercise tracking via WebSocket
    """
    try:
        print(f"Stopping exercise for session {request.sid}")
        
        if request.sid in active_sessions:
            session_data = active_sessions[request.sid]
            if 'stop_event' in session_data:
                session_data['stop_event'].set()
            if 'cap' in session_data and session_data['cap'] is not None:
                session_data['cap'].release()
            
        emit('exercise_stopped')
        
    except Exception as e:
        print(f"Error stopping exercise: {str(e)}")
        emit('error', {'message': f'Error stopping exercise: {str(e)}'})

def process_exercise_frames(session_id, exercise_id, stop_event):
    """
    Process exercise frames and send them via WebSocket
    
    Args:
        session_id: WebSocket session ID
        exercise_id: ID of the exercise to track
        stop_event: Event to signal when to stop processing
    """
    try:
        print(f"Processing exercise frames for {exercise_id}, session {session_id}")
        
        # Initialize video capture with optimized settings
        cap = cv2.VideoCapture(0)
        
        # Configure camera for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Failed to open camera")
            socketio.emit('error', {'message': 'Failed to open camera'}, room=session_id)
            return
        
        # Update session data
        active_sessions[session_id]['cap'] = cap
        
        # Initial variables
        left_counter = 0
        right_counter = 0
        left_state = None
        right_state = None
        frame_count = 0
        last_frame_time = time.time()
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                socketio.emit('error', {'message': 'Failed to capture camera frame'}, room=session_id)
                break
            
            # Process only every 2nd frame for better performance
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Exercise variables
            form_feedback = ""
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Define arm landmarks for exercise tracking
                arm_sides = {
                    'left': {
                        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                        'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                        'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                        'hip': mp_pose.PoseLandmark.LEFT_HIP
                    },
                    'right': {
                        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                        'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                        'hip': mp_pose.PoseLandmark.RIGHT_HIP
                    }
                }
                
                # Track angles and exercise state
                for side, joints in arm_sides.items():
                    shoulder = [
                        landmarks[joints['shoulder'].value].x,
                        landmarks[joints['shoulder'].value].y,
                    ]
                    elbow = [
                        landmarks[joints['elbow'].value].x,
                        landmarks[joints['elbow'].value].y,
                    ]
                    wrist = [
                        landmarks[joints['wrist'].value].x,
                        landmarks[joints['wrist'].value].y,
                    ]
                    
                    # Calculate elbow angle
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Display angle on frame
                    cv2.putText(
                        image,
                        f'{int(elbow_angle)}',
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Exercise specific logic - using hammer curl as an example
                    if exercise_id == 'hummer':
                        if side == 'left':
                            if elbow_angle > 160:
                                left_state = 'down'
                            if elbow_angle < 30 and left_state == 'down':
                                left_state = 'up'
                                left_counter += 1
                                form_feedback = "جيد! استمر"
                        
                        if side == 'right':
                            if elbow_angle > 160:
                                right_state = 'down'
                            if elbow_angle < 30 and right_state == 'down':
                                right_state = 'up'
                                right_counter += 1
                                form_feedback = "ممتاز! استمر"
                
                # Display counters on frame
                cv2.putText(image, f'Left: {left_counter}', (10, 50), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right: {right_counter}', (10, 100), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Update session counters
                active_sessions[session_id]['left_counter'] = left_counter
                active_sessions[session_id]['right_counter'] = right_counter
                
                # Display FPS on frame
                cv2.putText(image, f'FPS: {int(fps)}', (10, image.shape[0] - 20), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            # Convert frame to base64 for WebSocket transmission
            # Use lower quality JPEG compression for faster transmission
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, buffer = cv2.imencode('.jpg', image, encode_param)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame and data
            socketio.emit('exercise_frame', {
                'frame': frame_data,
                'left_counter': left_counter,
                'right_counter': right_counter,
                'feedback': form_feedback,
                'fps': int(fps)
            }, room=session_id)
            
            # Short delay to reduce CPU usage
            time.sleep(0.03)  # ~30 fps
        
        # Clean up camera when done
        if cap.isOpened():
            cap.release()
        
        print(f"Exercise processing stopped for session {session_id}")
        
    except Exception as e:
        print(f"Error in process_exercise_frames: {str(e)}")
        traceback.print_exc()
        socketio.emit('error', {'message': f'Error processing exercise: {str(e)}'}, room=session_id)
        
        # Cleanup
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            if 'cap' in session_data and session_data['cap'] is not None:
                session_data['cap'].release()

@socketio.on('send_frame')
def handle_frame(data):
    """
    Process frames sent from client and return real-time feedback
    """
    if request.sid not in active_sessions:
        emit('error', {'message': 'No active session'})
        return
        
    try:
        print(f"Received frame from {request.sid}, frame size: {len(data.get('frame', ''))} bytes")
        session_data = active_sessions[request.sid]
        exercise_id = session_data.get('exercise_id')
        
        # Decode base64 frame
        frame_data = data.get('frame')
        if not frame_data:
            emit('error', {'message': 'No frame data received'})
            return
            
        # Convert base64 to numpy array
        import numpy as np
        import cv2
        
        imgdata = base64.b64decode(frame_data)
        nparr = np.frombuffer(imgdata, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        result = process_frame(frame, exercise_id, session_data)
        
        # Send back the results
        emit('exercise_frame', {
            'left_counter': result.get('left_counter', 0),
            'right_counter': result.get('right_counter', 0),
            'feedback': result.get('feedback', ''),
            'frame': result.get('frame', '')
        })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        emit('error', {'message': f'Error processing frame: {str(e)}'})

def process_frame(frame, exercise_id, session_data):
    """
    Process a single frame for exercise tracking
    This function integrates with your existing exercise tracking code
    
    Args:
        frame: The video frame as numpy array
        exercise_id: ID of the exercise being performed
        session_data: Active session data containing counters and state
    
    Returns:
        Dictionary with tracking results
    """
    try:
        # Flip the frame horizontally for better user experience
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get current counter values from session
        left_counter = session_data.get('left_counter', 0)
        right_counter = session_data.get('right_counter', 0)
        left_state = session_data.get('left_state')
        right_state = session_data.get('right_state')
        
        # Default feedback
        form_feedback = ""
        
        if results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            # Define arm landmarks for exercise tracking
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST
                }
            }
            
            # Process both arms
            for side, joints in arm_sides.items():
                # Check if landmarks exist
                if (landmarks[joints['shoulder'].value].visibility > 0.5 and
                    landmarks[joints['elbow'].value].visibility > 0.5 and
                    landmarks[joints['wrist'].value].visibility > 0.5):
                    
                    # Extract coordinates
                    shoulder = [
                        landmarks[joints['shoulder'].value].x,
                        landmarks[joints['shoulder'].value].y,
                    ]
                    elbow = [
                        landmarks[joints['elbow'].value].x,
                        landmarks[joints['elbow'].value].y,
                    ]
                    wrist = [
                        landmarks[joints['wrist'].value].x,
                        landmarks[joints['wrist'].value].y,
                    ]
                    
                    # Calculate elbow angle
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Display angle on frame
                    cv2.putText(
                        image,
                        f'{int(elbow_angle)}',
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Exercise specific logic based on exercise_id
                    if exercise_id == 'hummer':
                        if side == 'left':
                            if elbow_angle > 160:
                                left_state = 'down'
                            if elbow_angle < 30 and left_state == 'down':
                                left_state = 'up'
                                left_counter += 1
                                form_feedback = "جيد! استمر"
                        
                        if side == 'right':
                            if elbow_angle > 160:
                                right_state = 'down'
                            if elbow_angle < 30 and right_state == 'down':
                                right_state = 'up'
                                right_counter += 1
                                form_feedback = "ممتاز! استمر"
                    
                    # Add other exercises based on your existing implementations
                    elif exercise_id == 'front_raise':
                        # Implement front raise logic
                        pass
                    elif exercise_id == 'squat':
                        # Implement squat logic
                        pass
                    # Add more exercises as needed
        
        # Update session data
        session_data['left_counter'] = left_counter
        session_data['right_counter'] = right_counter
        session_data['left_state'] = left_state
        session_data['right_state'] = right_state
        
        # Display counters on frame
        cv2.putText(image, f'Left: {left_counter}', (10, 50), 
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right: {right_counter}', (10, 100), 
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Convert processed frame to base64 if needed
        ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'left_counter': left_counter,
            'right_counter': right_counter,
            'feedback': form_feedback,
            'frame': processed_frame  # Return the processed frame
        }
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        traceback.print_exc()
        return {
            'left_counter': session_data.get('left_counter', 0),
            'right_counter': session_data.get('right_counter', 0),
            'feedback': f'Error: {str(e)}',
            'frame': ''
        }

# @socketio.on('process_frame')
# def handle_process_frame(data):
#     """
#     WebSocket handler for processing frames in real-time.
#     Accepts base64-encoded image frames and returns processing results.
#     """
#     try:
#         # Extract exercise type and image data
#         exercise_type = data.get('exercise_type', 'hummer')
#         img_data = data.get('image')
        
#         if not img_data:
#             emit('process_result', {"error": "No image data provided"})
#             return
            
#         # Decode base64 image
#         img_bytes = base64.b64decode(img_data.split(',')[1] if ',' in img_data else img_data)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None or frame.size == 0:
#             emit('process_result', {"error": "Invalid image format"})
#             return
        
#         # Process the frame
#         results = process_single_frame(frame, exercise_type)
        
#         # Send results back
#         emit('process_result', results)
        
#     except Exception as e:
#         app.logger.error(f"WebSocket error: {str(e)}")
#         app.logger.error(traceback.format_exc())
#         emit('process_result', {"error": str(e)})

@app.route('/api/exercise/start/<exercise>', methods=['POST'])
def start_exercise_api(exercise):
    """
    REST API endpoint to start an exercise
    Returns a session ID that can be used to retrieve frames
    """
    try:
        valid_exercises = get_valid_exercises()
        
        if exercise not in valid_exercises:
            return jsonify({"error": f"Invalid exercise: {exercise}"}), 400
            
        # Generate a unique session ID
        session_id = f"api_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Create a stop event to allow safe termination
        stop_event = threading.Event()
        
        # Store session data
        active_sessions[session_id] = {
            'exercise_id': exercise,
            'stop_event': stop_event,
            'cap': None,
            'left_counter': 0,
            'right_counter': 0,
            'start_time': time.time(),
            'last_access': time.time(),
            'frames': []  # Store last few frames
        }
        
        # Start a thread to process the exercise
        exercise_thread = threading.Thread(
            target=process_exercise_frames_api,
            args=(session_id, exercise, stop_event)
        )
        exercise_thread.daemon = True
        exercise_thread.start()
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "message": f"Exercise {exercise} started",
            "access_url": f"/api/exercise/frames/{session_id}"
        })
        
    except Exception as e:
        app.logger.error(f"Error starting exercise API: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/exercise/frames/<session_id>', methods=['GET'])
def get_exercise_frames(session_id):
    """
    Retrieve the latest frame and data for a session
    """
    try:
        if session_id not in active_sessions:
            return jsonify({"error": "Invalid or expired session ID"}), 404
            
        session_data = active_sessions[session_id]
        session_data['last_access'] = time.time()
        
        # If no frames available yet
        if not session_data.get('frames'):
            return jsonify({
                "status": "pending",
                "message": "Waiting for frames",
                "left_counter": session_data.get('left_counter', 0),
                "right_counter": session_data.get('right_counter', 0)
            })
        
        # Get the latest frame
        latest_frame = session_data['frames'][-1]
        
        return jsonify({
            "status": "success",
            "frame": latest_frame.get('frame'),
            "left_counter": session_data.get('left_counter', 0),
            "right_counter": session_data.get('right_counter', 0),
            "feedback": latest_frame.get('feedback', ''),
            "fps": latest_frame.get('fps', 0),
            "timestamp": latest_frame.get('timestamp', time.time())
        })
        
    except Exception as e:
        app.logger.error(f"Error getting exercise frames: {str(e)}")
        return jsonify({"error": str(e)}), 500
def process_frame_for_exercise(frame_data, exercise_id, session_data):
    try:
        # Your existing code
        
        print(f"Processing frame for exercise {exercise_id}")
        result = your_model_function(frame_data, exercise_id)
        print(f"Model result: {result}")
        
        # Rest of code
        
        return result
    except Exception as e:
        print(f"Error in process_frame_for_exercise: {str(e)}")
        traceback.print_exc()
        return {'error': str(e)}

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Process a single image frame sent from a mobile device.
    This endpoint allows the Flutter app to send camera frames for processing.
    """
    try:
        # Get the exercise type from the request
        exercise_type = request.form.get('exercise_type', 'hummer')
        
        # Validate exercise type
        if exercise_type not in exercise_map:
            return jsonify({"error": f"Invalid exercise type: {exercise_type}"}), 400
            
        # Get the image from the request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "No image provided"}), 400
            
        # Convert to OpenCV format
        nparr = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Process the frame using the appropriate exercise function
        results = process_single_frame(frame, exercise_type)
        
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def process_single_frame(frame, exercise_type):
    """
    Process a single video frame for pose detection and exercise tracking.
    
    Args:
        frame: OpenCV image frame
        exercise_type: Type of exercise to track (e.g., 'hummer', 'squat')
        
    Returns:
        Dictionary with processed results including counters, feedback, and annotated image
    """
    # Initialize storage for exercise state if not already present
    if not hasattr(process_single_frame, 'exercise_states'):
        process_single_frame.exercise_states = {}
    
    # Get or create state for this exercise type
    if exercise_type not in process_single_frame.exercise_states:
        process_single_frame.exercise_states[exercise_type] = {
            'left_counter': 0,
            'right_counter': 0,
            'left_state': None,
            'right_state': None,
            'last_feedback': '',
            'last_time': time.time()
        }
    
    # Access the state for this exercise
    state = process_single_frame.exercise_states[exercise_type]
    
    # Convert to RGB for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Initialize response data
    response_data = {
        "left_counter": state['left_counter'],
        "right_counter": state['right_counter'],
        "feedback": state['last_feedback'],
        "angles": {},
        "landmarks": {}
    }
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Draw pose landmarks on frame
        annotated_frame = draw_pose_landmarks(frame, results)
        
        # Extract key landmark coordinates for the specific exercise
        arm_sides = {
            'left': {
                'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                'hip': mp_pose.PoseLandmark.LEFT_HIP
            },
            'right': {
                'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                'hip': mp_pose.PoseLandmark.RIGHT_HIP
            }
        }
        
        # Process based on exercise type
        if exercise_type == 'hummer':
            # Process bicep curl (hummer) exercise
            for side, joints in arm_sides.items():
                # Extract joint coordinates
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y
                ]
                
                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Store angle in response
                response_data["angles"][f"{side}_elbow"] = elbow_angle
                
                # Rep counting logic for bicep curl
                if side == 'left':
                    if elbow_angle > 160:
                        state['left_state'] = 'down'
                    if elbow_angle < 30 and state['left_state'] == 'down':
                        state['left_state'] = 'up'
                        state['left_counter'] += 1
                        state['last_feedback'] = "Good form on left arm!"
                if side == 'right':
                    if elbow_angle > 160:
                        state['right_state'] = 'down'
                    if elbow_angle < 30 and state['right_state'] == 'down':
                        state['right_state'] = 'up'
                        state['right_counter'] += 1
                        state['last_feedback'] = "Good form on right arm!"
                
                # Draw angle on the annotated frame
                elbow_pixel = (
                    int(elbow[0] * annotated_frame.shape[1]),
                    int(elbow[1] * annotated_frame.shape[0])
                )
                cv2.putText(
                    annotated_frame,
                    f"{int(elbow_angle)}°",
                    elbow_pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
        elif exercise_type == 'squat':
            # Process squat exercise - simplified example
            leg_sides = {
                'left': {
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'ankle': mp_pose.PoseLandmark.LEFT_ANKLE
                },
                'right': {
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
                }
            }
            
            for side, joints in leg_sides.items():
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]
                knee = [
                    landmarks[joints['knee'].value].x,
                    landmarks[joints['knee'].value].y
                ]
                ankle = [
                    landmarks[joints['ankle'].value].x,
                    landmarks[joints['ankle'].value].y
                ]
                
                knee_angle = calculate_angle(hip, knee, ankle)
                response_data["angles"][f"{side}_knee"] = knee_angle
                
                # Basic squat detection
                if side == 'left' and knee_angle < 90:
                    state['left_state'] = 'down'
                elif side == 'left' and knee_angle > 160 and state['left_state'] == 'down':
                    state['left_state'] = 'up'
                    state['left_counter'] += 1
                    state['last_feedback'] = "Good squat!"
        
        # Add more exercise types following the same pattern
        # Each exercise would have its own angle calculations and rep counting logic
        
        # Store counters in response
        response_data["left_counter"] = state['left_counter']
        response_data["right_counter"] = state['right_counter']
        response_data["feedback"] = state['last_feedback']
        
        # Convert landmarks to a serializable format
        response_data["landmarks"] = {
            str(i): {"x": landmark.x, "y": landmark.y, "z": landmark.z, "visibility": landmark.visibility}
            for i, landmark in enumerate(landmarks)
        }
        
        # Convert the annotated frame to base64 for response
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        response_data["annotated_image"] = img_str
    
    return response_data

def draw_pose_landmarks(frame, results):
    """
    Draw pose landmarks and connections on the frame.
    
    Args:
        frame: Original image frame
        results: MediaPipe pose detection results
        
    Returns:
        Frame with landmarks and connections drawn
    """
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        annotated_frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
    # Add counter text
    if hasattr(process_single_frame, 'exercise_states'):
        for exercise_type, state in process_single_frame.exercise_states.items():
            cv2.putText(
                annotated_frame,
                f"Left: {state['left_counter']} | Right: {state['right_counter']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # Add feedback if available
            if state['last_feedback']:
                cv2.putText(
                    annotated_frame,
                    state['last_feedback'],
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    
    return annotated_frame

# @socketio.on('process_frame')
# def handle_process_frame(data):
#     """
#     WebSocket handler for processing frames in real-time.
#     Accepts base64-encoded image frames and returns processing results.
#     """
#     try:
#         # Extract exercise type and image data
#         exercise_type = data.get('exercise_type', 'hummer')
#         img_data = data.get('image')
        
#         if not img_data:
#             emit('process_result', {"error": "No image data provided"})
#             return
            
#         # Decode base64 image
#         img_bytes = base64.b64decode(img_data.split(',')[1] if ',' in img_data else img_data)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None or frame.size == 0:
#             emit('process_result', {"error": "Invalid image format"})
#             return
        
#         # Process the frame
#         results = process_single_frame(frame, exercise_type)
        
#         # Send results back
#         emit('process_result', results)
        
#     except Exception as e:
#         app.logger.error(f"WebSocket error: {str(e)}")
#         app.logger.error(traceback.format_exc())
#         emit('process_result', {"error": str(e)})

# @socketio.on('process_frame')
# def handle_process_frame(data):
#     try:
#         # Extract exercise type and image data
#         exercise_type = data.get('exercise_type', 'hummer')
#         img_data = data.get('image')
        
#         if not img_data:
#             emit('process_result', {"error": "No image data provided"})
#             return
            
#         # Decode base64 image
#         img_bytes = base64.b64decode(img_data.split(',')[1] if ',' in img_data else img_data)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Process the frame
#         results = process_single_frame(frame, exercise_type)
        
#         # Send results back
#         emit('process_result', results)
        
#     except Exception as e:
#         emit('process_result', {"error": str(e)})
# Fixed WebSocket handler for processing frames
@socketio.on('process_frame')
def handle_process_frame(data):
    """
    WebSocket handler for processing frames in real-time.
    Accepts base64-encoded image frames and returns processing results.
    """
    try:
        # Extract exercise type and image data
        exercise_type = data.get('exercise_type', 'hummer')
        img_data = data.get('image')
        
        if not img_data:
            emit('process_result', {"error": "No image data provided"})
            return
            
        # Log receiving frame data
        print(f"Received process_frame event for {exercise_type}, image size: {len(img_data) if img_data else 'none'}")
        
        # Decode base64 image
        try:
            # Handle both formats: with data URL prefix or without
            if ',' in img_data:
                img_bytes = base64.b64decode(img_data.split(',')[1])
            else:
                img_bytes = base64.b64decode(img_data)
                
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                emit('process_result', {"error": "Invalid image format or empty frame"})
                return
                
            print(f"Successfully decoded image: {frame.shape}")
        except Exception as decode_error:
            print(f"Error decoding image: {str(decode_error)}")
            emit('process_result', {"error": f"Error decoding image: {str(decode_error)}"})
            return
        
        # Process the frame
        results = process_single_frame(frame, exercise_type)
        
        # Send results back
        print(f"Sending process_result with data: {len(results.get('annotated_image', '')) if 'annotated_image' in results else 'no image'} bytes")
        emit('process_result', results)
        
    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in process_frame handler: {error_message}")
        print(stack_trace)
        emit('process_result', {"error": error_message, "details": stack_trace})
        
@app.route('/api/exercise/stop/<session_id>', methods=['POST'])
def stop_exercise_api(session_id):
    """
    Stop an exercise session
    """
    try:
        if session_id not in active_sessions:
            return jsonify({"error": "Invalid or expired session ID"}), 404
            
        session_data = active_sessions[session_id]
        if 'stop_event' in session_data:
            session_data['stop_event'].set()
        if 'cap' in session_data and session_data['cap'] is not None:
            session_data['cap'].release()
            
        # Keep session data for a short time for final frame retrieval
        # It will be cleaned up by the housekeeping task
        
        return jsonify({
            "status": "success",
            "message": f"Exercise stopped for session {session_id}",
            "left_counter": session_data.get('left_counter', 0),
            "right_counter": session_data.get('right_counter', 0)
        })
        
    except Exception as e:
        app.logger.error(f"Error stopping exercise: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_exercise_frames_api(session_id, exercise_id, stop_event):
    """
    Process exercise frames for the REST API
    
    Args:
        session_id: Session ID for tracking
        exercise_id: ID of the exercise to track
        stop_event: Event to signal when to stop processing
    """
    try:
        print(f"Processing API exercise frames for {exercise_id}, session {session_id}")
        
        # Initialize video capture with optimized settings
        cap = cv2.VideoCapture(0)
        
        # Configure camera for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Failed to open camera")
            active_sessions[session_id]['error'] = "Failed to open camera"
            return
        
        # Update session data
        active_sessions[session_id]['cap'] = cap
        
        # Initial variables
        left_counter = 0
        right_counter = 0
        left_state = None
        right_state = None
        frame_count = 0
        last_frame_time = time.time()
        
        # Max frames to store
        MAX_STORED_FRAMES = 5
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Process only every 3rd frame for better performance in API mode
            frame_count += 1
            if frame_count % 3 != 0:
                continue
                
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Convert to RGB for mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Exercise variables
            form_feedback = ""
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Define arm landmarks for exercise tracking
                arm_sides = {
                    'left': {
                        'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                        'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                        'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                        'hip': mp_pose.PoseLandmark.LEFT_HIP
                    },
                    'right': {
                        'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                        'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                        'hip': mp_pose.PoseLandmark.RIGHT_HIP
                    }
                }
                
                # Simplified exercise logic for API mode
                for side, joints in arm_sides.items():
                    shoulder = [
                        landmarks[joints['shoulder'].value].x,
                        landmarks[joints['shoulder'].value].y,
                    ]
                    elbow = [
                        landmarks[joints['elbow'].value].x,
                        landmarks[joints['elbow'].value].y,
                    ]
                    wrist = [
                        landmarks[joints['wrist'].value].x,
                        landmarks[joints['wrist'].value].y,
                    ]
                    
                    # Calculate elbow angle
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Display angle on frame
                    cv2.putText(
                        image,
                        f'{int(elbow_angle)}',
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Exercise specific logic - basic version
                    if exercise_id == 'hummer':
                        if side == 'left':
                            if elbow_angle > 160:
                                left_state = 'down'
                            if elbow_angle < 30 and left_state == 'down':
                                left_state = 'up'
                                left_counter += 1
                                form_feedback = "جيد! استمر"
                        
                        if side == 'right':
                            if elbow_angle > 160:
                                right_state = 'down'
                            if elbow_angle < 30 and right_state == 'down':
                                right_state = 'up'
                                right_counter += 1
                                form_feedback = "ممتاز! استمر"
                
                # Display counters on frame
                cv2.putText(image, f'Left: {left_counter}', (10, 50), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right: {right_counter}', (10, 100), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Display FPS
                cv2.putText(image, f'FPS: {int(fps)}', (10, image.shape[0] - 20), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Update session counters
                active_sessions[session_id]['left_counter'] = left_counter
                active_sessions[session_id]['right_counter'] = right_counter
            
            # Convert frame to base64 for API transmission
            # Use higher compression for API
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            ret, buffer = cv2.imencode('.jpg', image, encode_param)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Store frame data
            frame_info = {
                'frame': frame_data,
                'feedback': form_feedback,
                'fps': int(fps),
                'timestamp': time.time()
            }
            
            # Add to frames list, keeping only the last MAX_STORED_FRAMES
            active_sessions[session_id]['frames'].append(frame_info)
            if len(active_sessions[session_id]['frames']) > MAX_STORED_FRAMES:
                active_sessions[session_id]['frames'] = active_sessions[session_id]['frames'][-MAX_STORED_FRAMES:]
            
            # Longer delay for API mode to reduce server load
            time.sleep(0.1)  # ~10 fps
        
        # Clean up camera when done
        if cap.isOpened():
            cap.release()
        
        print(f"API exercise processing stopped for session {session_id}")
        
    except Exception as e:
        print(f"Error in process_exercise_frames_api: {str(e)}")
        traceback.print_exc()
        active_sessions[session_id]['error'] = str(e)
        
        # Cleanup
        if 'cap' in active_sessions[session_id] and active_sessions[session_id]['cap'] is not None:
            active_sessions[session_id]['cap'].release()

def cleanup_sessions():
    """
    Periodically clean up inactive sessions
    """
    while True:
        try:
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, session_data in active_sessions.items():
                # Check for API sessions that haven't been accessed in a while
                if session_id.startswith('api_') and 'last_access' in session_data:
                    if current_time - session_data['last_access'] > 60:  # 60 seconds timeout
                        sessions_to_remove.append(session_id)
                        if 'stop_event' in session_data:
                            session_data['stop_event'].set()
                        if 'cap' in session_data and session_data['cap'] is not None:
                            session_data['cap'].release()
            
            # Remove inactive sessions
            for session_id in sessions_to_remove:
                del active_sessions[session_id]
                print(f"Cleaned up inactive session: {session_id}")
                
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            
        # Sleep for 10 seconds before next cleanup
        time.sleep(10)

# Route for mobile-optimized view
@app.route('/mobile/<exercise>')
def mobile_exercise(exercise):
    """
    Mobile-optimized endpoint for exercise viewing
    """
    valid_exercises = get_valid_exercises()
    
    if exercise not in valid_exercises:
        app.logger.error(f"Invalid exercise requested: {exercise}")
        return "Exercise not found", 404
        
    return render_template('direct_video_fast.html', exercise_id=exercise)

# Route for debug view
@app.route('/debug/<exercise>')
def debug_exercise(exercise):
    """
    Debug endpoint for exercise viewing
    """
    valid_exercises = get_valid_exercises()
    
    if exercise not in valid_exercises:
        app.logger.error(f"Invalid exercise requested: {exercise}")
        return "Exercise not found", 404
        
    return render_template('direct_video_debug.html', exercise_id=exercise)

# Health monitoring endpoints
@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    """
    # Get basic application health metrics
    memory_usage = os.popen('ps -o rss= -p %d' % os.getpid()).read()
    if memory_usage:
        memory_usage = int(memory_usage.strip()) / 1024  # Convert to MB
    else:
        memory_usage = "Unknown"
    
    # Count active sessions
    active_count = len(active_sessions)
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app.config.get('START_TIME', time.time()),
        "active_sessions": active_count,
        "memory_usage_mb": memory_usage
    })
@app.route('/socket-test')
def socket_test():
    return render_template('websocket-test.html')
@app.route('/test')
def test():
    return "Hello, the app is working!"
# Simplified lightweight endpoint for mobile clients
@app.route('/api/exercise_simple/<exercise>', methods=['GET'])
def simple_exercise_stream(exercise):
    """
    Returns a simple MJPEG stream with minimal overhead for low-bandwidth clients
    """
    try:
        if exercise in exercise_map:
            # Use a generator with lower quality frames
            def generate_low_quality_frames():
                cap = cv2.VideoCapture(0)
                
                # Configure for lower quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                
                if not cap.isOpened():
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\n'
                           b'Camera not available\r\n')
                    return
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Skip frames for performance
                        for _ in range(2):
                            cap.read()
                            
                        # Resize to even smaller for bandwidth reduction
                        frame = cv2.resize(frame, (320, 240))
                        
                        # Lower quality JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                        frame_data = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                finally:
                    cap.release()
            
            # Return the stream with cache control headers
            return Response(
                generate_low_quality_frames(), 
                mimetype='multipart/x-mixed-replace; boundary=frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        else:
            return "Invalid exercise", 400
    except Exception as e:
        app.logger.error(f"Error in simple_exercise_stream: {str(e)}")
        app.logger.error(traceback.format_exc())
        return "Error processing video", 500

if __name__ == '__main__':
    # Initialize mediapipe
    try:
        import mediapipe as mp
        print(f"Mediapipe loaded successfully")
        
        # Initialize pose
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Medium complexity for balance between performance and accuracy
        )
        print("Pose model initialized successfully")
    except Exception as e:
        print(f"Error initializing libraries: {e}")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_sessions)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Set start time for uptime tracking
    app.config['START_TIME'] = time.time()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Print startup information
    print(f"Starting AI Fitness Trainer on port {port}")
    print(f"Available exercises: {', '.join(get_valid_exercises())}")
    
    # Run the application
    #app.run(host='0.0.0.0', port=port)
    socketio.run(app, host='0.0.0.0', port=port)