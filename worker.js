self.onmessage = async function(event) {
    const { appsScriptUrl, photo1Base64, photo2Base64 } = event.data;
    
    try {
        console.log('[Worker] Starting submission...');
        
        // This is a common workaround for Google Apps Script CORS issues.
        // It sends the data as plain text to avoid a complex "preflight" request
        // that browsers would otherwise send.
        const fetchOptions = {
            method: 'POST',
            mode: 'cors', // Explicitly set the mode
            headers: {
                'Content-Type': 'text/plain;charset=utf-8',
            }
        };

        // Step 1: Extract Data
        console.log('[Worker] Sending image for extraction...');
        const extractResponse = await fetch(appsScriptUrl, {
            ...fetchOptions,
            body: JSON.stringify({ action: 'extract', photo1Base64 }),
        });

        const extractResult = await extractResponse.json();
        if (!extractResult.success) {
            throw new Error(`Extraction failed: ${extractResult.message}`);
        }
        console.log('[Worker] Extraction successful.');
        const extractedData = extractResult.data;

        // Step 2: Save Data
        console.log('[Worker] Sending extracted data to be saved...');
        const saveResponse = await fetch(appsScriptUrl, {
            ...fetchOptions,
            body: JSON.stringify({
                action: 'save',
                extractedData: extractedData,
                photo1Base64: photo1Base64,
                photo2Base64: photo2Base64
            }),
        });

        const saveResult = await saveResponse.json();
        if (!saveResult.success) {
            throw new Error(`Save failed: ${saveResult.message}`);
        }
        console.log('[Worker] Submission process completed successfully.');

    } catch (err) {
        // This log will show up in the browser's developer console if something goes wrong.
        console.error('[Worker] An error occurred during the background submission:', err);
    }
};
