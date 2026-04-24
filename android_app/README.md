# Android app wrapper for Solar Forecast

This folder is a minimal Android Studio project that wraps your web app in a WebView.

## Run locally

1. Keep backend running:
   - `uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload`
2. Open `android_app` in Android Studio.
3. For emulator, `MainActivity.kt` already uses `http://10.0.2.2:8000/`.
4. For real phone, replace URL with your machine IP or deployed HTTPS URL.

## Build shareable APK

1. In Android Studio: `Build` -> `Build Bundle(s) / APK(s)` -> `Build APK(s)`.
2. APK output is under:
   - `android_app/app/build/outputs/apk/debug/app-debug.apk`

## Before publishing

- Deploy backend to HTTPS URL.
- Update `webView.loadUrl(...)` to that URL.
- Configure proper app icon and signing.

