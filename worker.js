self.onmessage = async function(event) {
    const { appsScriptUrl, photo1Base64, photo2Base64 } = event.data;
    
    try {
        console.log('[Worker] Starting submission...');
        
        // --- THIS IS THE FIX ---
        // By changing the Content-Type, we avoid the complex security check
        // that your script is not set up to handle. Your script can still
        // read the data perfectly fine this way.

        // Step 1: Extract Data
        const extractResponse = await fetch(appsScriptUrl, {
            method: 'POST',
            body: JSON.stringify({ action: 'extract', photo1Base64 }),
            headers: { 'Content-Type': 'text/plain' } // Changed from application/json
        });

        const extractResult = await extractResponse.json();
        if (!extractResult.success) {
            throw new Error(`Extraction failed: ${extractResult.message}`);
        }
        console.log('[Worker] Extraction successful.');
        const extractedData = extractResult.data;

        // Step 2: Save Data
        const saveResponse = await fetch(appsScriptUrl, {
            method: 'POST',
            body: JSON.stringify({
                action: 'save',
                extractedData: extractedData,
                photo1Base64: photo1Base64,
                photo2Base64: photo2Base64
            }),
            headers: { 'Content-Type': 'text/plain' } // Changed from application/json
        });

        const saveResult = await saveResponse.json();
        if (!saveResult.success) {
            throw new Error(`Save failed: ${saveResult.message}`);
        }
        console.log('[Worker] Submission process completed successfully.');

    } catch (err) {
        console.error('[Worker] An error occurred during the background submission:', err);
    }
};
