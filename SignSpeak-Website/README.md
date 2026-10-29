# SignSpeak — Hand Gestures to Words

A no-database, static website using the webcam and MediaPipe's pretrained Gesture Recognizer. It runs in the browser and can be hosted on GitHub Pages.

## Files
- `index.html` — page structure
- `style.css` — responsive dark UI
- `app.js` — webcam, gesture recognition, phrase builder and text-to-speech

## Run locally
Camera permissions generally require HTTPS or `localhost`.
1. Extract the files.
2. Open a terminal in this folder.
3. Run `python -m http.server 8000`
4. Visit `http://localhost:8000`

## Deploy on GitHub Pages
1. Create a GitHub repository.
2. Upload `index.html`, `style.css`, `app.js`, and this README to the repository root.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Choose `main` and `/ (root)`, then save.
6. Open the published HTTPS link and allow camera access.

## Important limitations
This is a learning prototype, not a complete sign-language translator. It maps a small number of general hand gestures to convenient words. General gesture labels are not equivalent to sign-language signs. Full sign-language translation requires a suitable sign-language dataset/model and often needs motion, both hands, facial expressions and grammar. The app does not use a database and does not upload or store camera video. It does download the MediaPipe runtime/model from public CDNs, so internet access is needed.
