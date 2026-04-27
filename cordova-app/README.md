# PPE Detection Edge Device Mobile App

This is the mobile application for the PPE Detection system, built using Apache Cordova. It allows users to perform real-time PPE detection using their mobile device's camera, connecting to the backend server.

## Features

- **Camera Integration**: Use device camera for real-time PPE detection
- **File Upload**: Upload images or videos for analysis
- **Server Configuration**: Easily configure the backend server URL
- **Mobile Optimized**: Responsive UI designed for mobile devices
- **Offline Capable**: Basic functionality works offline, syncs when connected

## Prerequisites

- Node.js and npm (for Cordova CLI)
- Java JDK 11+ (for Android builds)
- Android Studio with Android SDK (for APK generation)

## Setup

### 1. Install Cordova CLI (if not already installed)
```bash
npm install -g cordova
```

### 2. Install Dependencies
```bash
cd cordova-app
npm install
```

### 3. Add Android Platform
```bash
cordova platform add android
```

## Building the APK

### 1. Ensure Android SDK is configured
Set environment variables:
- `JAVA_HOME`: Path to JDK installation
- `ANDROID_HOME`: Path to Android SDK (usually `~/Android/Sdk`)

### 2. Build the APK
```bash
cordova build android
```

The APK will be generated at:
`platforms/android/app/build/outputs/apk/debug/app-debug.apk`

## Installation on Device

1. Transfer the APK file to your Android device
2. Enable "Install from unknown sources" in device settings
3. Install the APK
4. Launch the app

## Configuration

### Server URL Setup
1. Open the app
2. Go to the "Settings" tab
3. Enter your server URL (e.g., `http://192.168.1.100:8000`)
4. Tap "Save & Test"

### Finding Server IP
On Windows (host machine):
```cmd
ipconfig
```
Look for "IPv4 Address" under your network adapter.

## Usage

### Camera Mode
- Switch to "Camera" tab
- Grant camera permissions when prompted
- Tap "Start Camera"
- Point at subjects to detect PPE
- Use "Capture & Detect" for single shots or "Start Live" for continuous detection

### Upload Mode
- Switch to "Upload" tab
- Tap the upload area to select image/video files
- The app will process and display results
- View → Tool Windows → Device File Explorer
- During testing - upload file into Android studio manually
- View → Tool Windows → Device File Explorer -> /storage/emulated/0/Download -> right click -> upload

### Settings
- Configure server connection
- View current server URL

## Backend Requirements

The mobile app requires a running PPE Detection backend server. Ensure:

- Backend is running on accessible IP/port
- CORS is configured to allow mobile app connections
- Network firewall allows connections from mobile device

## Troubleshooting

### Connection Issues
- Verify server URL in Settings tab
- Check if backend is running and accessible
- Ensure mobile device is on same network as server

### Camera Not Working
- Grant camera permissions in device settings
- Restart the app after granting permissions

### Build Issues
- Verify JDK and Android SDK versions
- Check environment variables are set correctly
- Ensure Cordova and platforms are up to date

## Development

### Modifying the App
- Edit `www/index.html` for UI changes
- Use React for component updates
- Test changes with `cordova run android` (requires device/emulator)

### Adding Plugins
```bash
cordova plugin add <plugin-name>
```

## License

This project is part of the PPE Detection Synergy Project.</content>
<parameter name="filePath">d:\NaishadhStudy\Sem3\project\ppe-detection-synergy-project\cordova-app\README.md