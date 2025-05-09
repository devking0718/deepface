from deepface import DeepFace
import os
import cv2
import numpy as np

def detect_and_show_faces(image_path):
    try:
        # Verify the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image")
            
        # Extract faces from the image
        faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False  # Set to True if you want strict face detection
        )
        
        # Draw rectangles around detected faces
        for face in faces:
            if 'facial_area' in face:
                area = face['facial_area']
                x, y = area['x'], area['y']
                w, h = area['w'], area['h']
                
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add confidence score if available
                if 'confidence' in face:
                    confidence = f"{face['confidence']:.2f}"
                    cv2.putText(img, confidence, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                              (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return faces
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    image_path = "img1.jpg"
    
    faces = detect_and_show_faces(image_path)
    
    if faces is not None:
        print(f"Found {len(faces)} faces in the image")
        for i, face in enumerate(faces, 1):
            print(f"\nFace #{i}:")
            if 'confidence' in face:
                print(f"Confidence: {face['confidence']:.2f}")
            if 'facial_area' in face:
                area = face['facial_area']
                print(f"Location: x={area['x']}, y={area['y']}")
                print(f"Width: {area['w']}, Height: {area['h']}")
    else:
        print("No faces detected or an error occurred")










