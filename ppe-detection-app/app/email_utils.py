import smtplib
from email.message import EmailMessage
import cv2
import time
import os 
from dotenv import load_dotenv

load_dotenv()

# CONFIG (move to env later)
SENDER_EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
RECEIVER_EMAIL = os.getenv("EMAIL_TO")

import threading

def _send_email_worker(frame_copy):
    if not all([SENDER_EMAIL, APP_PASSWORD, RECEIVER_EMAIL]):
        print("Cannot send email: MISSING environment variables (SENDER_EMAIL, APP_PASSWORD, or RECEIVER_EMAIL)")
        return

    try:
        import uuid
        filename = f"violation_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("/tmp", filename)
        cv2.imwrite(filepath, frame_copy)

        msg = EmailMessage()
        msg['Subject'] = "PPE Violation Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        msg.set_content(
           "A PPE violation has been detected.\n\n"
            "Please find the attached image."
        )

        with open(filepath, 'rb') as f:
            msg.add_attachment(
                f.read(),
                maintype='image',
                subtype='jpeg',
                filename = filename
            )
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("Email alert sent successfully")

        # Cleanup 
        os.remove(filepath)

    except Exception as e:
        print(f"Email sending failed: {e}")

def send_email_alert(frame):
    """
    Spins off the email sending to a separate thread so it doesn't block processing.
    """
    # Create a copy so the main thread can continue without corrupting this frame
    frame_copy = frame.copy()
    thread = threading.Thread(target=_send_email_worker, args=(frame_copy,))
    thread.start()
