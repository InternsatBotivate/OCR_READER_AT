// This is the full content for worker.js
self.onmessage = async function(event) {
    // --- [UPDATED] ---
    const { ocrBackendUrl, photo1Base64, photo2Base64 } = event.data;
    // --- [END OF UPDATE] ---
    
    try {
        self.postMessage({ status: 'info', message: 'Worker starting submission to: ' + ocrBackendUrl });
        console.log('[Worker] Starting submission to:', ocrBackendUrl);

        const response = await fetch(ocrBackendUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // --- [UPDATED] ---
            // Send both images to the backend
            body: JSON.stringify({ 
                base64Image1: photo1Base64,
                base64Image2: photo2Base64 
            }),
            // --- [END OF UPDATE] ---
        });

        if (!response.ok) {
            let errorMsg = `Server error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorMsg = `Request failed: ${errorData.detail || response.statusText}`;
            } catch (e) {
                // Ignore if response wasn't JSON
            }
            throw new Error(errorMsg);
        }

        const ocrResult = await response.json();
        console.log('[Worker] OCR and Validation successful:', ocrResult);
        
        self.postMessage({ status: 'success', data: ocrResult });

    } catch (err) {
        console.error('[Worker] An error occurred:', err.message);
        self.postMessage({ status: 'error', message: err.message });
    }
};