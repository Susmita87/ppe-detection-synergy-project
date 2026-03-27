import smtplib
from email.message import EmailMessage
import cv2
import time
import os 

# 🔹 CONFIG (move to env later)
SENDER_EMAIL = os.getenv("EMAIL_USER")
#APP_PASSWORD = "fktc qjan umhs gerw"
SENDER_EMAIL = os.getenv("EMAIL_PASS")
RECEIVER_EMAIL = os.getenv("EMAIL_TO")

def send_email_alert(frame):
    try:
        filename = f"violation_{int(time.time())}.jpg"
        filepath = os.path.join("/tmp", filename)
        cv2.imwrite(filepath, frame)

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

        print("📧 Email alert sent successfully")

        # Cleanup 
        os.remove(filepath)

    except Exception as e:
        print(f"Email sending failed: {e}")